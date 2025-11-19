import os
from thinker.util import __version__
from thinker.util import __project__

import cv2
import argparse
import numpy as np
import copy
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn.functional as F
import textwrap
from thinker.main import Env
from thinker.util import init_env_out, create_env_out
from thinker.actor_net import ActorNet
import thinker.util as util
import gym

def plot_gym_env_out(x, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(
        np.transpose(x, (1, 2, 0)),
        interpolation="nearest",
        aspect="auto",
    )
    if title is not None:
        ax.set_title(title)


def plot_multi_gym_env_out(xs, titles=None, col_n=5):
    size_n = 7
    row_n = (len(xs) + (col_n - 1)) // col_n

    fig, axs = plt.subplots(row_n, col_n, figsize=(col_n * size_n, row_n * size_n))
    if len(axs.shape) == 1:
        axs = axs[np.newaxis, :]
    m = 0
    for y in range(row_n):
        for x in range(col_n):
            if m >= len(xs):
                axs[y][x].set_axis_off()
            else:
                axs[y][x].imshow(np.transpose(xs[m], (1, 2, 0)))
                axs[y][x].set_title(
                    "rollout %d" % (m + 1) if titles is None else titles[m]
                )
            m += 1
    plt.tight_layout()
    return fig


def plot_policies(logits, labels, action_meanings, ax=None, title="Real policy prob"):
    if ax is None:
        fig, ax = plt.subplots()
    probs = []
    for logit, k in zip(logits, labels):
        if k not in ["model policy", "action"]:
            probs.append(torch.softmax(logit, dim=-1).detach().cpu().numpy())
        else:
            probs.append(logit.detach().cpu().numpy())

    ax.set_title(title)
    xs = np.arange(len(probs[0]))
    for n, (prob, label) in enumerate(zip(probs, labels)):
        ax.bar(xs + 0.1 * (n - len(logits) // 2), prob, width=0.1, label=label)
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(logits[0].shape[-1])))
    ax.set_xticklabels(action_meanings, rotation=90)
    plt.subplots_adjust(bottom=0.2)
    ax.set_ylim(0, 1)
    ax.legend()


def plot_base_policies(prob, action_meanings, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    rec_t, num_actions = prob.shape
    xs = np.arange(rec_t)
    labels = action_meanings
    for i in range(num_actions):
        c = ax.bar(
            xs + 0.8 * (i / num_actions),
            prob[:, i],
            width=0.8 / (num_actions),
            label=labels[i],
        )
        color = c.patches[0].get_facecolor()
        color = color[:3] + (color[3] * 0.5,)
        ax.bar(
            xs + 0.8 * (i / num_actions),
            prob[:, i],
            width=0.8 / (num_actions),
            color=color,
        )
    ax.legend()
    ax.set_ylim(0, 1)
    ax.set_title("Model policy prob")


def plot_im_policies(
    pri_logits,
    reset_logits,
    pri,
    cur_reset,
    action_meanings,
    one_hot=True,
    reset_ind=0,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    rec_t, dim_actions, num_actions = pri_logits.shape
    pri_logits = pri_logits[:10, 0]
    pri = pri[:10, 0]    
    reset_logits = reset_logits[:10]
    cur_reset = cur_reset[:10]

    num_actions += 1
    rec_t -= 1

    im_prob = torch.softmax(pri_logits, dim=-1).detach().cpu().numpy()
    reset_prob = (
        torch.softmax(reset_logits, dim=-1)[:, [reset_ind]]
        .detach()
        .cpu()
        .numpy()
    )
    full_prob = np.concatenate([im_prob, reset_prob], axis=-1)

    if not one_hot:
        pri = F.one_hot(pri, num_actions - 1)
    pri = pri.detach().cpu().numpy()
    cur_reset = cur_reset.unsqueeze(-1).detach().cpu().numpy()
    full_action = np.concatenate([pri, cur_reset], axis=-1)

    xs = np.arange(pri_logits.shape[0])
    labels = action_meanings.copy()
    labels.append("cur_reset")

    for i in range(num_actions):
        c = ax.bar(
            xs + 0.8 * (i / num_actions),
            full_prob[:, i],
            width=0.8 / (num_actions),
            label=labels[i],
        )
        color = c.patches[0].get_facecolor()
        color = color[:3] + (color[3] * 0.5,)
        ax.bar(
            xs + 0.8 * (i / num_actions),
            full_action[:, i],
            width=0.8 / (num_actions),
            color=color,
        )
    ax.legend()
    ax.set_ylim(0, 1)
    ax.set_title("Imagainary policy prob")

def plot_qn_sa(q_s_a, n_s_a, action_meanings, max_q_s_a=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    xs = np.arange(len(q_s_a))

    ax.bar(xs - 0.3, q_s_a.cpu(), color="g", width=0.3, label="q_s_a")
    ax_n = ax.twinx()
    if max_q_s_a is not None:
        ax.bar(xs, max_q_s_a.cpu(), color="r", width=0.3, label="max_q_s_a")
    ax_n.bar(
        xs + (0.3 if max_q_s_a is not None else 0.0),
        n_s_a.cpu(),
        bottom=0,
        color="b",
        width=0.3,
        label="n_s_a",
    )
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(q_s_a))))
    ax.set_xticklabels(action_meanings, rotation=90)
    plt.subplots_adjust(bottom=0.2)
    ax.legend(loc="lower left")
    ax_n.legend(loc="lower right")
    ax.set_title("q_s_a and n_s_a")


def _extract_cur_reward(info):
    if info is None or "cur_reward" not in info or info["cur_reward"] is None:
        return np.nan

    value = info["cur_reward"]
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1)
        return value[0].item() if value.numel() > 0 else np.nan

    value = np.asarray(value)
    if value.size == 0:
        return np.nan
    return float(value.reshape(-1)[0])

def _collect_encoder_vectors(env, env_out, device):
    real_vectors = None
    im_vectors = None
    im_vp_vectors = None
    im_vectors_from_real = False

    if not hasattr(env, "model_net"):
        return real_vectors, im_vectors, im_vp_vectors, im_vectors_from_real

    sr_net = getattr(env.model_net, "sr_net", None)
    if sr_net is not None and env_out.real_states is not None:
        with torch.no_grad():
            real_state_input = env_out.real_states[0, 0, :].unsqueeze(0).unsqueeze(0)
            real_state_norm = env.model_net.normalize(real_state_input)
            dummy_action = torch.zeros(
                1, 1, sr_net.dim_rep_actions, device=device
            )
            dummy_done = torch.zeros(1, 1, dtype=torch.bool, device=device)
            real_vectors, _ = sr_net.encoder(
                real_state_norm, dummy_done, dummy_action, {}, flatten=True
            )
            real_vectors = real_vectors.cpu().numpy()
            if hasattr(sr_net, "state") and isinstance(sr_net.state, dict) and "sr_h" in sr_net.state:
                im_vectors = sr_net.state["sr_h"]
                if torch.is_tensor(im_vectors):
                    im_vectors = im_vectors.detach().cpu().numpy()
            if im_vectors is None:
                im_vectors = real_vectors
                im_vectors_from_real = True

    vp_net = getattr(env.model_net, "vp_net", None)
    if vp_net is not None and hasattr(vp_net, "state") and isinstance(vp_net.state, dict) and "vp_h" in vp_net.state:
        im_vp_vectors = vp_net.state["vp_h"]
        if torch.is_tensor(im_vp_vectors):
            im_vp_vectors = im_vp_vectors.detach().cpu().numpy()

    return real_vectors, im_vectors, im_vp_vectors, im_vectors_from_real

def _append_encoder_vectors(video_stats, real_vectors, im_vectors, im_vp_vectors, status_value, shared_source):
    if status_value == 0 and shared_source:
        real_vectors = None
    video_stats["real_vectors"].append(real_vectors)
    video_stats["im_vectors"].append(im_vectors)
    video_stats["im_vp_vectors"].append(im_vp_vectors)


def gen_video(video_stats, file_path):
    import cv2

    _, real_h, real_w = video_stats["real_imgs"][0].shape
    _, im_h, im_w  = video_stats["im_imgs"][0].shape
    need_resacle = real_h != im_h or real_w != im_w    

    for l in range(len(video_stats["real_imgs"])):
        video_stats["real_imgs"][l] = np.transpose((video_stats["real_imgs"][l]), (1, 2, 0))          

    for l in range(len(video_stats["im_imgs"])):
        im_img = np.transpose((video_stats["im_imgs"][l]), (1, 2, 0))
        im_img = np.clip(im_img, 0, 1) * 255
        im_img = im_img.astype(np.uint8)  
        if need_resacle:          
            im_img = cv2.resize(im_img, (real_w, real_h), interpolation=cv2.INTER_NEAREST)
        video_stats["im_imgs"][l] = im_img

    # Generate video
    imgs = []
    h, w, c = video_stats["real_imgs"][0].shape

    for i in range(len(video_stats["real_imgs"])):
        img = np.zeros(shape=(h, w * 2, 3), dtype=np.uint8)
        real_img = np.copy(video_stats["real_imgs"][i])
        im_img = np.copy(video_stats["im_imgs"][i])
        if c == 1:
            real_img = np.repeat(real_img, 3, axis=2)
            im_img = np.repeat(im_img, 3, axis=2)   
            
            # reset; yellow tint
            im_img[:, :, 0] = 255 * 0.3 + im_img[:, :, 0] * 0.7
            im_img[:, :, 1] = 255 * 0.3 + im_img[:, :, 1] * 0.7
        elif video_stats["status"][i] == 3:
            # force reset; red tint
            im_img[:, :, 0] = 255 * 0.3 + im_img[:, :, 0] * 0.7
        elif video_stats["status"][i] == 0:
            # real reset; blue tint
            im_img[:, :, 2] = 255 * 0.3 + im_img[:, :, 2] * 0.7

        img[:, :w] = real_img
        img[:, w:] = im_img
        imgs.append(img)

    enlarge_fcator = 3
    width = w * 2 * enlarge_fcator
    height = h * enlarge_fcator
    fps = 5

    path = os.path.join(file_path, "video.avi")
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    video = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
    video.set(cv2.CAP_PROP_BITRATE, 10000)  # set the video bitrate to 10000 kb/s
    for img in imgs:
        height, width, _ = img.shape
        new_height, new_width = height * enlarge_fcator, width * enlarge_fcator
        img = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)
    video.release()


def print_im_actions(im_dict, action_meanings, real_action, print_stat=False):
    lookup_dict = {k: v for k, v in enumerate(action_meanings)}
    print_strs = []
    n, s = 1, ""
    reset = False

    def a_to_str(a):
        if a.dim() >= 1:
            a = a.tolist()
            return "(" + ",".join([f"{num:.2f}" for num in a]) + ")"
        else:
            return lookup_dict[a.item()]
    for im, reset in zip(im_dict["pri"][:-1], im_dict["cur_reset"][:-1]):        
        s += a_to_str(im) + ", "
        if reset:
            s += "cur_reset" if reset == 1 else "FReset"
            print_strs.append("%d: %s" % (n, s))
            s = ""
            n += 1
    if not reset:
        print_strs.append("%d: %s" % (n, s[:-2]))
    
    print_strs.append("Real action: %s" % a_to_str(real_action))
    if print_stat:
        for s in print_strs:
            print(s)
    return print_strs


def save_concatenated_image(buf1, buf2, strs, outdir, height=2500, width=3000):
    # Render the first figure onto a PIL image
    buf1.seek(0)
    img1 = Image.open(buf1)
    margin1 = 0

    # Render the second figure onto a PIL image
    buf2.seek(0)
    img2 = Image.open(buf2)
    margin2 = 0.1

    # Resize the images to have the desired width
    w1, h1 = img1.size
    new_width = int(width * (1 - 2 * margin1))
    new_height = int((new_width / w1) * h1)
    img1 = img1.resize((new_width, new_height))

    w2, h2 = img2.size
    new_width = int(width * (1 - 2 * margin2))
    new_height = int((new_width / w2) * h2)
    img2 = img2.resize((new_width, new_height))

    # Create a new image with the desired size
    result = Image.new("RGB", (width, height), color="white")

    # Paste the first image on the top
    result.paste(im=img1, box=(int(width * margin1), 0))

    # Paste the second image below the first one
    result.paste(im=img2, box=(int(width * margin2), img1.height))

    # Add the long string below the second image
    draw = ImageDraw.Draw(result)
    font_path = os.path.join(cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf")
    #font_path = os.path.join("C:\\", "Windows", "Fonts", "calibri.ttf")
    font = ImageFont.truetype(font_path, 34)

    y = (
        img1.height + img2.height + 10
    )  # Leave some space between the second image and the text
    font_box = font.getbbox("A")  # Get the height of a typical line of text
    line_height = font_box[3] - font_box[1]

    margint = 0.1
    for line in strs:
        # Wrap the line to fit the width of the image
        wrapper = textwrap.TextWrapper(width=int(width * (1 - 2 * margint)))
        lines = wrapper.wrap(line)

        for wrapped_line in lines:
            draw.text((width * margint, y), wrapped_line, font=font, fill="black")
            y += line_height

        # Add extra line spacing between paragraphs
        y += line_height
    # Save the concatenated image
    result.save(outdir)


def visualize(
    savedir,
    xpid,
    outdir,
    plot=False,
    saveimg=True,
    savevideo=True,
    seed=-1,
    max_frames=-1,
    use_gpu=True,  # GPU 사용 여부 추가
    save_encoder_vectors=True,  # encoder 벡터 저장 옵션 추가
):        
    savedir = savedir.replace("__project__", __project__)
    ckpdir = os.path.join(savedir, xpid)      
    if os.path.islink(ckpdir): ckpdir = os.readlink(ckpdir)  
    ckpdir =  os.path.abspath(os.path.expanduser(ckpdir))
    outdir = os.path.abspath(os.path.expanduser(outdir))

    max_eps_n = 1
    config_path = os.path.join(ckpdir, 'config_c.yaml')
    flags = util.create_flags(config_path, save_flags=False)
    if seed < 0:
        seed = np.random.randint(10000)
    
    # GPU 사용 설정
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for visualization")
    else:
        device = torch.device("cpu")
        print("Using CPU for visualization")
    
    env = Env(
        name=flags.name,
        env_n=1,
        base_seed=seed,        
        gpu=use_gpu,  # GPU 사용 설정
        train_model=False,
        parallel=False,
        savedir=savedir,        
        xpid=xpid,
        ckp=True,
        return_x=True,
        )
    
    render = "Safexp" in flags.name

    if "Sokoban" in flags.name:
        action_meanings = ["NOOP", "UP", "DOWN", "LEFT", "RIGHT"]
    elif flags.sample_n > 0:
        action_meanings = [str(n) for n in range(flags.sample_n)]
    else:
        action_meanings = [str(n) for n in range(env.num_actions)]
    num_actions = env.num_actions

    print("Sampled seed: %d" % seed)    

    obs_space = env.observation_space
    action_space = env.action_space  
    actor_param = {
                "obs_space":obs_space,
                "action_space":action_space,
                "flags":flags,
                "tree_rep_meaning": env.get_tree_rep_meaning(),
            }

    actor_net = ActorNet(**actor_param)
    checkpoint = torch.load(
        os.path.join(ckpdir, "ckp_actor.tar"), device, weights_only = False
    )
    actor_net.set_weights(checkpoint["actor_net_state_dict"])
    actor_net.to(device)  # GPU로 이동
    actor_state = actor_net.initial_state(batch_size=1)
    if use_gpu:
        actor_state = tuple(s.to(device) if s is not None else None for s in actor_state)
    
    print("Actor Net Real Steps: %d Steps: %d" % (checkpoint["real_step"],
                                                  checkpoint["step"])
                                                  )
    # create output folder
    n = 0
    while True:
        name = "%s-%d-%d" % (flags.xpid, checkpoint["real_step"], n)
        outdir_ = os.path.join(outdir, name)
        if not os.path.exists(outdir_):
            os.makedirs(outdir_)
            print(f"Outputting to {outdir_}")
            break
        n += 1
    outdir = outdir_

    # initalize env
    state, info = env.reset()
    env_out = init_env_out(state, info, flags, actor_net.dim_actions, actor_net.tuple_action)
    
    # GPU로 데이터 이동
    if use_gpu:
        env_out = env_out._replace(
            xs=env_out.xs.to(device),
            real_states=env_out.real_states.to(device),
            tree_reps=env_out.tree_reps.to(device),
            episode_return=env_out.episode_return.to(device),
            done=env_out.done.to(device),
            real_done=env_out.real_done.to(device),
            step_status=env_out.step_status.to(device)
        )
    
    # some initial setting
    plt.rcParams.update({"font.size": 15})

    tree_reps = env.decode_tree_reps(env_out.tree_reps)
    end_gym_env_outs, end_titles = [], []
    ini_max_q = tree_reps["max_rollout_return"][0, 0].item()

    step = 0
    real_step = 0
    returns, model_policy = (
        [],
        [],
    )
    im_list = ["pri_logits", "reset_logits", "pri", "cur_reset"]
    im_dict = {k: [] for k in im_list}
    im_done = False

    # video_stats 초기화 - 기존 이미지와 encoder 벡터 모두 저장
    video_stats = {"real_imgs": [], "im_imgs": [], "status": [], "tree_reps": [], "cur_rewards": []}
    if save_encoder_vectors:
        video_stats.update({"real_vectors": [], "im_vectors": [], "im_vp_vectors": []})
        print("Saving both images and encoder vectors")
    
    # 메모리 효율성을 위한 최대 저장 프레임 수 제한 (전체 프레임 저장을 위해 충분히 큰 값 설정)
    max_video_frames = 10000000000000  # 최대 100,000프레임까지 저장 가능

    if flags.grayscale and "Sokoban" not in flags.name:
        copy_n = 1
    else:
        copy_n = 3

    if not render:
        root_real_states = env_out.real_states[0, 0, -copy_n:].cpu().numpy() 
        last_root_real_states = root_real_states
        root_xs = env_out.xs[0, 0, -copy_n:].cpu().numpy()
    else:
        root_real_states = env.render(mode='rgb_array', camera_id=0)[0] 
    
    video_stats["real_imgs"].append(root_real_states)
    video_stats["im_imgs"].append(root_xs)
    
        # SRN latent output 저장 (옵션) - real_imgs/im_imgs와 동일한 로직으로 저장
    if save_encoder_vectors:
        status_value = 0
        real_vectors, im_vectors, im_vp_vectors, vectors_from_real = _collect_encoder_vectors(
            env, env_out, device
        )
        _append_encoder_vectors(
            video_stats,
            real_vectors,
            im_vectors,
            im_vp_vectors,
            status_value,
            vectors_from_real,
        )

    video_stats["status"].append(0)  # 0 for real step, 1 for reset, 2 for normal
    video_stats["tree_reps"].append({k: v.cpu().numpy() for k, v in tree_reps.items()})
    video_stats["cur_rewards"].append(_extract_cur_reward(info))

    # 메모리 관리를 위한 배치 크기 설정
    batch_size = 5  # 한 번에 처리할 프레임 수 제한 (더 작게 설정)
    
    while len(returns) < max_eps_n:
        step += 1
        
        # 메모리 정리
        if step % batch_size == 0:
            if use_gpu:
                torch.cuda.empty_cache()
                # GPU 메모리 사용량 출력
                if step % (batch_size * 5) == 0:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"Step {step}: GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                    print(f"Video frames stored: {len(video_stats['real_imgs'])}")
        
        actor_out, actor_state = actor_net(env_out, actor_state)        
        action = actor_out.action

        last_real_step = (env_out.step_status == 0) | (env_out.step_status == 3)

        if last_real_step:
            agent_v = actor_out.baseline[0, 0, 0]

        # additional stat record - GPU에서 CPU로 이동하여 저장
        im_dict["pri_logits"].append(actor_out.pri_param[:,0].cpu())
        im_dict["reset_logits"].append(actor_out.reset_logits[:,0].cpu())
        im_dict["pri"].append(actor_out.pri[:,0].cpu())
        im_dict["cur_reset"].append(actor_out.reset[:,0].cpu())
        
        tree_reps_ = env.decode_tree_reps(env_out.tree_reps)
        model_policy.append(tree_reps_["cur_policy"].cpu())       

        state, reward, done, truncated_done, info = env.step(action[0], action[1])
        last_real_step = (info["step_status"] == 0) | (info["step_status"] == 3)
        next_real_step = (info["step_status"] == 2) | (info["step_status"] == 3)

        if render:
            if not last_real_step:
                if flags.sample_n > 0:
                    cur_raw_action = (tree_reps_["cur_raw_action"].view(flags.sample_n, env.raw_dim_actions)*env.raw_num_actions)[action[0][0]]
                    cur_raw_action = cur_raw_action.long().unsqueeze(0)
                else:
                    cur_raw_action = action[0]
                if not im_done: _, _, im_done, _ = env.unwrapped_step(cur_raw_action.cpu().numpy())

        env_out = create_env_out(action, state, reward, done, truncated_done, info, flags)
        
        # GPU로 데이터 이동
        if use_gpu:
            env_out = env_out._replace(
                xs=env_out.xs.to(device),
                real_states=env_out.real_states.to(device),
                tree_reps=env_out.tree_reps.to(device),
                episode_return=env_out.episode_return.to(device),
                done=env_out.done.to(device),
                real_done=env_out.real_done.to(device),
                step_status=env_out.step_status.to(device)
            )

        tree_reps = env.decode_tree_reps(env_out.tree_reps)
        if (
            len(im_dict["cur_reset"]) > 0
            and tree_reps["cur_reset"]
            and not actor_out.reset
        ):
            im_dict["cur_reset"][-1] = im_dict["cur_reset"][-1].clone()
            im_dict["cur_reset"][-1][:] = 3  # force reset

        if not render:
            #img = env.unnormalize(torch.clamp(env_out.xs, 0, 1)).to(torch.uint8)
            xs = torch.clamp(env_out.xs, 0, 1)
            xs = xs[0, 0, -copy_n:].cpu().numpy()
        else:
            xs = env.render(mode='rgb_array', camera_id=0)[0]   

        if render and (tree_reps["cur_reset"] == 1 or next_real_step):
            im_done = False   

       # if ~last_real_step and (
       #     tree_reps["cur_reset"] == 1 or next_real_step
       # ):
        title = "pred v: %.2f" % (tree_reps["cur_v"][:, 0].item())
        title += " pred g: %.2f" % (tree_reps["rollout_return"][:, 0].item())
        title += " pred r: %.2f" % (tree_reps["cur_r"][:, 0].item())
        if len(end_gym_env_outs) < 10: end_gym_env_outs.append(xs)
        end_titles.append(title)

                # SRN latent output 저장 (옵션) - real_imgs/im_imgs와 동일한 로직으로 저장
        if save_encoder_vectors:
            real_vectors, im_vectors, im_vp_vectors, vectors_from_real = _collect_encoder_vectors(
                env, env_out, device
            )
        else:
            real_vectors = None
            im_vectors = None
            im_vp_vectors = None
            vectors_from_real = False

        cur_reward_value = _extract_cur_reward(info)

        # record data for generating video (전체 프레임 저장)
        if len(video_stats["real_imgs"]) < max_video_frames:
            if last_real_step:
                root_real_states = env_out.real_states[0, 0, -copy_n:].cpu().numpy() 
                root_xs = xs
                # real action            
                status_value = 0
            else:
                # imagainary action
                status_value = 2
            video_stats["status"].append(status_value)
            video_stats["real_imgs"].append(root_real_states)
            video_stats["im_imgs"].append(xs)
            
            # SRN encoder 벡터 저장
            if save_encoder_vectors:
                _append_encoder_vectors(
                    video_stats,
                    real_vectors,
                    im_vectors,
                    im_vp_vectors,
                    status_value,
                    vectors_from_real,
                )
            
            video_stats["tree_reps"].append(
                {k: v.cpu().numpy() for k, v in tree_reps.items()}
            )
            video_stats["cur_rewards"].append(cur_reward_value)

            if im_dict["cur_reset"][-1] in [1, 3]:
                # reset / force reset
                video_stats["real_imgs"].append(root_real_states)
                video_stats["im_imgs"].append(root_xs)
                reset_status = im_dict["cur_reset"][-1].item()
                if save_encoder_vectors:
                    _append_encoder_vectors(
                        video_stats,
                        real_vectors,
                        im_vectors,
                        im_vp_vectors,
                        reset_status,
                        vectors_from_real,
                    )
                video_stats["status"].append(reset_status)
                video_stats["tree_reps"].append(
                    {k: v.cpu().numpy() for k, v in tree_reps.items()}
                )
                video_stats["cur_rewards"].append(cur_reward_value)

        # visualize when a real step is made
        if (saveimg or plot) and last_real_step:
            fig, axs = plt.subplots(1, 5, figsize=(50, 10))
            for k in im_list:
                if im_dict[k][0] is not None:
                    im_dict[k] = torch.concat(im_dict[k], dim=0)
                else:
                    im_dict[k] = None
            plot_gym_env_out(last_root_real_states, axs[0], title="Real State")
            last_root_real_states = root_real_states
            plot_base_policies(
                torch.concat(model_policy)[:, :num_actions], action_meanings=action_meanings, ax=axs[1]
            )
            plot_im_policies(
                **im_dict,
                action_meanings=action_meanings,
                one_hot=False,
                reset_ind=1,
                ax=axs[2],
            )
            if "root_qs_mean" in tree_reps_.keys():
                plot_qn_sa(
                    q_s_a=tree_reps_["root_qs_mean"][0],
                    n_s_a=tree_reps_["root_ns"][0],
                    action_meanings=action_meanings,
                    max_q_s_a=tree_reps_["root_qs_max"][0],
                    ax=axs[3],
                )
            model_policy_logits = tree_reps_["root_policy"][0].view(actor_net.dim_actions, actor_net.num_actions)
            agent_policy_logits = actor_out.pri_param[0, 0]
            action = torch.nn.functional.one_hot(
                actor_out.pri[0, 0], env.num_actions
            )
            plot_policies(
                [model_policy_logits[0], agent_policy_logits[0], action[0]],
                ["model policy", "agent policy", "action"],
                action_meanings=action_meanings,
                ax=axs[4],
            )
            # plt.tight_layout()
            if saveimg:
                buf1 = BytesIO()
                plt.savefig(buf1, format="png")
            if plot:
                plt.show()
            plt.close()

            if len(end_gym_env_outs) > 0:
                plot_multi_gym_env_out(end_gym_env_outs, end_titles)
            if saveimg:
                buf2 = BytesIO()
                plt.savefig(buf2, format="png")
            if plot:
                plt.show()
            plt.close()

            log_str = "Step:%d (%d); return %.4f(%.4f) done %s real_done %s" % (
                real_step,
                step,
                env_out.episode_return[0, 0, 0],
                env_out.episode_return[0, 0, 1] if flags.im_cost > 0.0 else 0,
                "True" if env_out.done[0, 0] else "False",
                "True" if env_out.real_done[0, 0] else "False",
            )
            print(log_str)

            stat = f"Real Step: {real_step} Root v: {tree_reps_['root_v'][0, 0].item():.2f} Actor v: {agent_v:.2f}"
            stat += f" Root Max Q: {tree_reps_['max_rollout_return'][0, 0].item():.2f} Init. Root Max Q: {ini_max_q:.2f}"
            stat += f" Root Mean Q: {info['baseline'][0, 0].item():.2f}"
            if 'cur_reward' in info: stat += f" CurReward: {info['cur_reward'][0].item():.6f}"

            if flags.im_cost > 0.0:
                title += " im_return: %.4f" % env_out.episode_return[..., 1]

            if saveimg:
                im_action_strs = print_im_actions(
                    im_dict, action_meanings, actor_out.action[0][0], print_stat=plot
                )
                save_concatenated_image(
                    buf1,
                    buf2,
                    [stat] + im_action_strs,
                    os.path.join(outdir, f"{real_step}.png"),
                )
                buf1.close()
                buf2.close()

            im_dict = {k: [] for k in im_list}
            model_policy, end_gym_env_outs, end_titles = [], [], []
            ini_max_q = tree_reps["max_rollout_return"][0, 0].item()

            real_step += 1
            #if real_step >= 5: break

        if torch.any(env_out.real_done):
            step = 0
            new_rets = env_out.episode_return[env_out.real_done][:, 0].cpu().numpy()
            returns.extend(new_rets)
            print(
                "Finish %d episode: avg. return: %.2f (+-%.2f) "
                % (
                    len(returns),
                    np.average(returns),
                    np.std(returns) / np.sqrt(len(returns)),
                )
            )

        if max_frames >= 0 and real_step > max_frames:
            break

    if savevideo:
        video_stats["tree_reps"] = {
            k: np.concatenate([v[k] for v in video_stats["tree_reps"]], axis=0)
            for k in video_stats["tree_reps"][0].keys()
        }
        video_stats["cur_rewards"] = np.array(video_stats["cur_rewards"], dtype=np.float32)
        # gen_video(video_stats, outdir)
        np.save(os.path.join(outdir, "video_stat.npy"), video_stats)

    # 메모리 정리
    if use_gpu:
        torch.cuda.empty_cache()
    
    return video_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Thinker visualization")
    parser.add_argument("--outdir", default="../test", help="Output directory.")
    parser.add_argument("--savedir", default="../logs/__project__", 
                        help="Checkpoint directory.")
    parser.add_argument("--xpid", default="latest", help="id of the run.")    
    parser.add_argument("--project", default="", help="project of the run.")  
    parser.add_argument("--seed", default="-1", type=int, help="Base seed.")
    parser.add_argument(
        "--max_frames",
        default="-1",
        type=int,
        help="Max number of real frames to record",
    )
    parser.add_argument(
        "--use_gpu",
        default=True,
        type=bool,
        help="Whether to use GPU for visualization",
    )
    parser.add_argument(
        "--save_encoder_vectors",
        default=True,
        type=bool,
        help="Whether to save SRN encoder vectors in addition to images",
    )
    flags = parser.parse_args()    
    if flags.project: flags.savedir=flags.savedir.replace("__project__", flags.project)

    visualize(
        savedir=flags.savedir,
        xpid=flags.xpid,
        outdir=flags.outdir,
        plot=False,
        saveimg=True,
        savevideo=True,
        seed=flags.seed,
        max_frames=flags.max_frames,
        use_gpu=flags.use_gpu,
        save_encoder_vectors=flags.save_encoder_vectors,
    )
