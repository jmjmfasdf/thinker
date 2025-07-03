import json
import logging
import sys
import os
import random
import tempfile
import shutil
import numpy as np

# from scipy import misc
import imageio
import subprocess
import argparse

from SerializableStateObservation import SerializableStateObservation, Phase, Observation
from CompetitionParameters import CompetitionParameters
from ElapsedCpuTimer import ElapsedCpuTimer
from IOSocket import IOSocket
from Types import LEARNING_SSO_TYPE


class ClientCommGYM:
    """
     * Client communication, set up the socket for a given agent
    """

    def __init__(self, game, version, lvl, pathStr):
        self.tempDir = tempfile.TemporaryDirectory()
        self.addLevel('')   #Level template to be loaded into java

        self.TOKEN_SEP = '#'
        self.io = IOSocket(self.tempDir.name)
        self.sso = SerializableStateObservation()
        #self.agentName = agentName
        self.lastMessageId = 0
        self.LOG = False
        self.player = None
        self.global_ect = None
        self.lastSsoType = LEARNING_SSO_TYPE.JSON
        
        # Memory pooling for image processing
        self._image_buffer = np.zeros((84, 84, 3), dtype=np.uint8)
        self._temp_buffer = np.zeros((84, 84, 4), dtype=np.uint8)  # For RGBA conversion
        
        self.sso.Terminal=False

        baseDir = os.path.join(pathStr, 'gvgai')
        srcDir = os.path.join(baseDir, 'src')
        buildDir = os.path.join(baseDir, 'GVGAI_Build')
        gamesDir = os.path.join(pathStr, 'games', '{}_v{}'.format(game, version))
        cmd = ["java", "-Djava.awt.headless=true", "-classpath", buildDir, "tracks.singleLearning.utils.JavaServer", "-game", game, "-gamesDir", gamesDir, "-imgDir", baseDir, "-portNum", str(self.io.port)]

        #Check build version
        sys.path.append(baseDir)
        import check_build

        if(not os.path.isdir(buildDir)):
            raise Exception("Couldn't find build directory. Please run build.py from the install directory or reinstall with pip.")
        elif(not check_build.isCorrectBuild(srcDir, buildDir)):
            raise Exception("Your build is out of date. Please run build.py from the install directory or reinstall with pip.")
        else:
            try:
                self.java = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, cwd=self.tempDir.name)
            except subprocess.CalledProcessError as e:
                print('exit code: {}'.format(e.returncode))
                print('stderr: {}'.format(e.stderr.decode(sys.getfilesystemencoding())))

        self.startComm()
        self.reset(lvl)

    def _process_image_efficient(self, img_path):
        """
        Efficiently process image using pre-allocated buffers to minimize memory allocation
        """
        try:
            loaded_img = imageio.imread(img_path)
            
            # First handle resizing if needed
            if loaded_img.shape[:2] != (84, 84):
                from PIL import Image
                pil_img = Image.fromarray(loaded_img)
                pil_img = pil_img.resize((84, 84), Image.NEAREST)
                loaded_img = np.array(pil_img)
            
            # Now handle different image formats efficiently
            if len(loaded_img.shape) == 2:  # Grayscale
                # Use broadcasting instead of stack for memory efficiency
                self._image_buffer[:, :, 0] = loaded_img
                self._image_buffer[:, :, 1] = loaded_img
                self._image_buffer[:, :, 2] = loaded_img
            elif len(loaded_img.shape) == 3:
                if loaded_img.shape[2] == 4:  # RGBA
                    # Remove alpha channel in-place
                    self._image_buffer[:, :, :3] = loaded_img[:, :, :3]
                elif loaded_img.shape[2] == 1:  # Single channel
                    # Use broadcasting instead of repeat
                    single_channel = loaded_img[:, :, 0]
                    self._image_buffer[:, :, 0] = single_channel
                    self._image_buffer[:, :, 1] = single_channel
                    self._image_buffer[:, :, 2] = single_channel
                else:  # RGB or more channels
                    self._image_buffer[:, :, :3] = loaded_img[:, :, :3]
            
            # Ensure uint8 dtype (in-place if possible)
            if self._image_buffer.dtype != np.uint8:
                self._image_buffer = self._image_buffer.astype(np.uint8)
            
            return self._image_buffer.copy()
            
        except Exception as e:
            print(f"[WARNING] Failed to load/process image: {e}")
            # Return default color image
            self._image_buffer.fill(0)
            return self._image_buffer.copy()

    def startComm(self):
        self.io.initBuffers()
        #Reset currently sends initial commucations (which can't handle levels) and then resets
        #This should be split into two functions after the competition (July 18, 2018)
        self.reset(0)

    """
     * Method that perpetually listens for messages from the server.
     * With the use of additional helper methods, this function interprets
     * messages and represents the core response-generation methodology of the agent.
     * @throws IOException
    """

    def step(self,act):
        #self.sso.phase = Phase.ACT
        if not self.sso.Terminal:
            if(act == 0):
                self.act("")
            else:
                action = self.sso.availableActions[act - 1]
                self.act(action)
            self.line = self.io.readLine()
            self.line = self.line.rstrip("\r\n")
            self.processLine(self.line)
            
            score = self.reward()
            self.lastScore=self.sso.gameScore
        else:
            score=0
        
        if self.sso.isGameOver==True or self.sso.gameWinner=='PLAYER_WINS' or self.sso.phase == "FINISH" or self.sso.phase=="ABORT" or self.sso.phase=="End":
            # Use efficient image processing
            img_path = os.path.join(self.tempDir.name, 'gameStateByBytes.png')
            self.sso.image = self._process_image_efficient(img_path)
            self.sso.Terminal=True
        else:
            self.sso.Terminal=False
            actions=self.actions()
        
        info = {'winner': self.sso.gameWinner, 'actions': self.actions()}  
        return self.sso.image, score, self.sso.Terminal, info

    def reset(self, lvl):
        #flag=True
        #self.line = ''
        self.lastScore=0
        
        if hasattr(self,'line'):
            flag=True
            restart=True
            
            #self.io.writeToServer(self.lastMessageId, "END_TRAINING", self.LOG)
            
            #self.line = self.io.readLine()
            #self.line = self.line.rstrip("\r\n")
            #self.processLine(self.line)
            
            if self.sso.Terminal:
                self.io.writeToServer(self.lastMessageId, str(lvl) + "#" + self.lastSsoType, self.LOG)
            else:
            
                self.io.writeToServer(self.lastMessageId, "END_OVERSPENT", self.LOG)
            
                self.line = self.io.readLine()
                self.line = self.line.rstrip("\r\n")
                self.processLine(self.line)
            
                self.io.writeToServer(self.lastMessageId, str(lvl) + "#" + self.lastSsoType, self.LOG)

        else:
            restart=True
            flag=True
            #self.startComm()
            self.line = ''


        while flag:
            if restart:
                self.line = self.io.readLine()
                self.line = self.line.rstrip("\r\n")
                self.processLine(self.line)
            else:
                self.line=''

            if self.sso.phase == Phase.START:
                self.start()


            elif self.sso.phase == "INIT":
                self.sso.phase = Phase.INIT
                self.init()

            elif self.sso.phase == "ACT":
                flag=False                
                
                for i in range(1):
                    self.act(0)
                    self.line = self.io.readLine()
                    self.line = self.line.rstrip("\r\n")
                    self.processLine(self.line)

                if(self.sso.isGameOver==True or self.sso.gameWinner=='WINNER' or self.sso.phase == "FINISH" or self.sso.phase == "End"):
                    img_path = os.path.join(self.tempDir.name, 'gameStateByBytes.png')
                    # Use efficient image processing
                    self.sso.image = self._process_image_efficient(img_path)
                    self.sso.Terminal=True
                    self.lastScore=0
                else:
                    self.sso.Terminal=False
                

        return self.sso.image

    # def reward(self):
    #     scoreDelta = self.sso.gameScore-self.lastScore
    #     if(self.sso.gameWinner=='WINNER' or scoreDelta > 0):
    #         return 1
    #     elif(self.sso.isGameOver or scoreDelta < 0):
    #         return -1
    #     else:
    #         return 0

    def reward(self):
        scoreDelta = self.sso.gameScore-self.lastScore
        return scoreDelta

    def actions(self):
        nil = ["ACTION_NIL"]
        return nil + self.sso.availableActions

    def as_sso(self, d):
        self.sso.__dict__.update(d)
        return self.sso

    def parse_json(self, input):
        parsed_input = json.loads(input)
        self.sso.__dict__.update(parsed_input)
        if parsed_input.get('observationGrid'):
            self.sso.observationGrid = [[[None for j in range(self.sso.observationGridMaxCol)]
                                         for i in range(self.sso.observationGridMaxRow)]
                                        for k in range(self.sso.observationGridNum)]
            for i in range(self.sso.observationGridNum):
                for j in range(len(parsed_input['observationGrid'][i])):
                    for k in range(len(parsed_input['observationGrid'][i][j])):
                        self.sso.observationGrid[i][j][k] = Observation(parsed_input['observationGrid'][i][j][k])

        if parsed_input.get('NPCPositions'):
            self.sso.NPCPositions = [[None for j in
                                      range(self.sso.NPCPositionsMaxRow)] for i in
                                     range(self.sso.NPCPositionsNum)]
            for i in range(self.sso.NPCPositionsNum):
                for j in range(len(parsed_input['NPCPositions'][i])):
                    self.sso.NPCPositions[i][j] = Observation(parsed_input['NPCPositions'][i][j])

        if parsed_input.get('immovablePositions'):
            self.sso.immovablePositions = [[None for j in
                                            range(self.sso.immovablePositionsMaxRow)] for i in
                                           range(self.sso.immovablePositionsNum)]
            for i in range(self.sso.immovablePositionsNum):
                for j in range(len(parsed_input['immovablePositions'][i])):
                    self.sso.immovablePositions[i][j] = Observation(parsed_input['immovablePositions'][i][j])

        if parsed_input.get('movablePositions'):
            self.sso.movablePositions = [[None for j in
                                          range(self.sso.movablePositionsMaxRow)] for i in
                                         range(self.sso.movablePositionsNum)]
            for i in range(self.sso.movablePositionsNum):
                for j in range(len(parsed_input['movablePositions'][i])):
                    self.sso.movablePositions[i][j] = Observation(parsed_input['movablePositions'][i][j])

        if parsed_input.get('resourcesPositions'):
            self.sso.resourcesPositions = [[None for j in
                                            range(self.sso.resourcesPositionsMaxRow)] for i in
                                           range(self.sso.resourcesPositionsNum)]
            for i in range(self.sso.resourcesPositionsNum):
                for j in range(len(parsed_input['resourcesPositions'][i])):
                    self.sso.resourcesPositions[i][j] = Observation(parsed_input['resourcesPositions'][i][j])

        if parsed_input.get('portalsPositions'):
            self.sso.portalsPositions = [[None for j in
                                          range(self.sso.portalsPositionsMaxRow)] for i in
                                         range(self.sso.portalsPositionsNum)]
            for i in range(self.sso.portalsPositionsNum):
                for j in range(len(parsed_input['portalsPositions'][i])):
                    self.sso.portalsPositions[i][j] = Observation(parsed_input['portalsPositions'][i][j])

        if parsed_input.get('fromAvatarSpritesPositions'):
            self.sso.fromAvatarSpritesPositions = [[None for j in
                                                    range(self.sso.fromAvatarSpritesPositionsMaxRow)] for i in
                                                   range(self.sso.fromAvatarSpritesPositionsNum)]
            for i in range(self.sso.fromAvatarSpritesPositionsNum):
                for j in range(len(parsed_input['fromAvatarSpritesPositions'][i])):
                    self.sso.fromAvatarSpritesPositions[i][j] = Observation(parsed_input['fromAvatarSpritesPositions'][i][j])


    """
     * Method that interprets the received messages from the server's side.
     * A message can either be a string (in the case of initialization), or
     * a json object containing an encapsulated state observation.
     * This method deserializes the json object into a local state observation
     * instance.
     * @param msg Message received from server to be interpreted.
     * @throws IOException
    """

    def processLine(self, msg):
        try:
            if msg is None:
                print ("Message is null")
                return

            message = msg.split(self.TOKEN_SEP)
            if len(message) < 2:
                print ("Message not complete")
                return

            self.lastMessageId = message[0]
            js = message[1]

            self.sso = SerializableStateObservation()
            if js == "START":
                self.sso.phase = Phase.START
            elif js == "FINISH":
                self.sso.phase = Phase.FINISH
            else:
                js.replace('"', '')
                self.parse_json(js)
                # self.sso = json.loads(js, object_hook=self.as_sso)
            if self.sso.phase == "ACT":
                if(self.lastSsoType == LEARNING_SSO_TYPE.IMAGE or self.lastSsoType == "IMAGE" \
                        or self.lastSsoType == LEARNING_SSO_TYPE.BOTH or self.lastSsoType == "BOTH"):
                    if(self.sso.imageArray):
                        self.sso.convertBytesToPng(self.sso.imageArray, self.tempDir.name)
                        img_path = os.path.join(self.tempDir.name, 'gameStateByBytes.png')
                        # Use the optimized image processing function
                        self.sso.image = self._process_image_efficient(img_path)

        except Exception as e:
            logging.exception(e)
            print("Line processing [FAILED]")
            #traceback.print_exc()
            sys.exit()

    """
     * Manages the start of the communication. It starts the whole process, and sets up the timer for the whole run.
    """

    def start(self):
        self.global_ect = ElapsedCpuTimer()
        self.global_ect.setMaxTimeMillis(CompetitionParameters.TOTAL_LEARNING_TIME)
        ect = ElapsedCpuTimer()
        ect.setMaxTimeMillis(CompetitionParameters.START_TIME)
        #self.startAgent()
        if self.global_ect.exceededMaxTime():
            self.io.writeToServer(self.lastMessageId, "START_FAILED", self.LOG)
        else:
            self.io.writeToServer(self.lastMessageId, "START_DONE", self.LOG)

    """
     * Manages the init of a game played.
    """

    def init(self):
        ect = ElapsedCpuTimer()
        ect.setMaxTimeMillis(CompetitionParameters.INITIALIZATION_TIME)
        #self.player.init(self.sso, ect.copy())
        self.lastSsoType = LEARNING_SSO_TYPE.IMAGE
        if ect.exceededMaxTime():
            self.io.writeToServer(self.lastMessageId, "INIT_FAILED", self.LOG)
        else:
            self.io.writeToServer(self.lastMessageId, "INIT_DONE" + "#" + self.lastSsoType, self.LOG)

    """
     * Manages the action request for an agent. The agent is requested for an action,
     * which is sent back to the server
    """

    def act(self,action):
        ect = ElapsedCpuTimer()
        ect.setMaxTimeMillis(CompetitionParameters.ACTION_TIME)
        #act = self.player.act(self.sso, ect.copy())
        
        if (not action) or (action == ""):
            action = "ACTION_NIL"
        self.lastSsoType = LEARNING_SSO_TYPE.IMAGE

        if ect.exceededMaxTime() or ect.elapsedMillis() > CompetitionParameters.ACTION_TIME_DISQ:
            self.io.writeToServer(self.lastMessageId, "END_OVERSPENT", self.LOG)
        else:
            self.io.writeToServer(self.lastMessageId, str(action) + "#" + self.lastSsoType, self.LOG)

    def addLevel(self, path):
        lvlName = os.path.join(self.tempDir.name, 'game_lvl5.txt')
        if(path is ''):
            open(lvlName, 'w+').close()
        else:
            shutil.copyfile(path, lvlName)

    def __del__(self):
        try:
            self.java.kill()
        except:
            pass