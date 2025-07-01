from gymnasium.envs.registration import register
import os

dir_path = os.path.dirname(__file__)
games_path = os.path.join(dir_path, os.path.normpath('envs/games'))
games = os.listdir(games_path)

for game in games:
    game_path = os.path.join(games_path, game)
    if os.path.isdir(game_path):
        # JavaServer.java expects levels lvl0 - lvl4
        lvls = len([lvl for lvl in os.listdir(game_path) if 'lvl' in lvl])
        for lvl in range(lvls):
            # Extract game name and version
            name = game.split('_')[0]
            version = int(game.split('_')[-1][1:])

            register(
                id=f"gvgai-{name}-lvl{lvl}-v{version}",
                entry_point="gym_gvgai.envs.gvgai_env:GVGAI_Env",
                kwargs={
                    "game": name,
                    "level": lvl,
                    "version": version
                },
                max_episode_steps=2000,
            )