"""Guided one-command Spot session (spec section 11): tag, end, start, then the four arms, detached.

`python main.py` with no arguments runs `main()`; `python main.py --spot_sim` runs it against the sim.
The session connects without the lease, so the operator drives with the tablet during setup. Poses
are saved in the AprilTag's frame, so start and end survive reboots: every session re-detects the tag.

The arms run one after another under a detached runner, `python -m deploy.spot_session --run ROOT`,
which records each arm's status and run directories in `ROOT/session.json`.
"""
import datetime
import glob
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys

from envs.spot_mapless import RunFrame

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ('sharsa', 'dual', 'phys', 'geo')
ARM_FLAGS = {
    'sharsa': ['--agent=agents/sharsa.py'],
    'dual': [
        '--agent=agents/sharsa_dual.py', '--agent.rep_type=bilinear', '--agent.goalrep_dim=256',
        '--agent.rep_expectile=0.9', '--agent.rep_w=1.0',
    ],
    'phys': [
        '--agent=agents/sharsa.py', '--agent.w_fk=1.0', '--agent.viscous_scale=0.01', '--agent.num_walks=10',
        '--agent.fk_kappa=15.0',
    ],
    'geo': [
        '--agent=agents/sharsa_geodesic.py', '--agent.use_anisotropic=False', '--agent.w_geo=1.0',
        '--agent.kappa=16.7', '--agent.use_fk_loss=False', '--agent.basic_smoothing=False', '--agent.cost_dims=2',
    ],
}  # fmt: skip
COMMON_FLAGS = [
    '--warmup_steps=1000', '--save_interval=1000', '--log_interval=250', '--max_episode_steps=300',
    '--buffer_size=200000', '--seed=1', '--env_name=spot-ab-v0',
    '--agent.gc_negative=True', '--agent.value_loss_type=squared', '--agent.subgoal_steps=10',
    '--agent.batch_size=256',
]  # fmt: skip

# Sim: a fake tag at the old grid's waypoint_0, facing +x. End defaults near waypoint_17; start is
# 0.5 m in front of the tag. Both are grid-frame poses.
SIM_TAG = (1.77, 0.085, 0.0)
SIM_END = (5.27, 5.71, 0.0)
SIM_START = (SIM_TAG[0] + 0.5, SIM_TAG[1], 0.0)

NOT_DETECTED = 'AprilTag not detected. Move Spot so the tag is in view and press Enter to retry.'


def load_session(path):
    with open(path) as f:
        return json.load(f)


def save_session(path, session):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(session, f, indent=2)
    os.replace(tmp, path)  # the runner and a resuming session never read a half-written file


def tag_frame(client, session, print_fn=print):
    """The tag frame now: the tag re-detected in the vision frame, else the pose the session stored."""
    found = client.detect_fiducial(session['tag_id'])
    if found is not None:
        print_fn(f'tag {found[0]} detected at {found[1]} (vision frame)')
        return RunFrame(found[1])
    print_fn(f'tag {session["tag_id"]} not in view: using the session pose {session["tag_pose"]}')
    return RunFrame(session['tag_pose'])


class SimSessionClient:
    """The sim's stand-in for the robot during setup: the tag is always seen at SIM_TAG."""

    def detect_fiducial(self, tag_id=None):
        return 0, SIM_TAG


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except (OSError, TypeError):
        return False
    return True


def latest_checkpoint(root, arm):
    """(run_dir, step) of the arm's newest saved buffer, or None. Params may be missing (warmup)."""
    best = None
    for path in glob.glob(os.path.join(root, 'runs', arm, '**', 'buffer_*.npz'), recursive=True):
        m = re.fullmatch(r'buffer_(\d+)\.npz', os.path.basename(path))
        if m and (best is None or int(m.group(1)) > best[1]):
            best = (os.path.dirname(path), int(m.group(1)))
    return best


def arm_command(root, arm, hostname, online_steps, restore=None):
    cmd = [
        sys.executable, 'main.py', '--mapless', '--goal_mode=ab',
        f'--session_path={os.path.join(root, "session.json")}', f'--robot_hostname={hostname}',
        f'--save_dir={os.path.join(root, "runs", arm)}', f'--run_group=spot_session_{arm}',
        f'--online_steps={online_steps}', ARM_FLAGS[arm][0], *COMMON_FLAGS, *ARM_FLAGS[arm][1:],
    ]  # fmt: skip  (absl needs --agent= before any --agent.x)
    if restore is not None:
        cmd += [f'--restore_path={restore[0]}', f'--restore_epoch={restore[1]}']
    return cmd


def run_arms(root, popen=subprocess.Popen, print_fn=print):
    """The detached runner: each unfinished arm in order, resuming from its last checkpoint.

    Stops at the first arm that exits non-zero. SIGINT to the process group (`kill -INT -- -PID`)
    reaches the arm too, which saves its buffer and weights; the runner waits for that, then stops.
    """
    path = os.path.join(root, 'session.json')
    session = load_session(path)
    session['runner_pid'] = os.getpid()
    save_session(path, session)
    online_steps = int(os.environ.get('SPOT_ONLINE_STEPS') or 18000)
    os.makedirs(os.path.join(root, 'logs'), exist_ok=True)
    for arm in session['arms']:
        record = session['runs'][arm]
        if record['status'] == 'done':
            continue
        restore = latest_checkpoint(root, arm)
        if restore is not None and restore[1] >= online_steps:
            record['status'] = 'done'
            save_session(path, session)
            continue
        cmd = arm_command(root, arm, session['hostname'], online_steps, restore)
        record['status'] = 'running'
        save_session(path, session)
        log_path = os.path.join(root, 'logs', f'{arm}.log')
        print_fn(f'=== {arm}: {" ".join(cmd)}\n    log: {log_path}', flush=True)
        stopped = False
        with open(log_path, 'a') as log:
            proc = popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            while True:
                try:
                    code = proc.wait()
                    break
                except KeyboardInterrupt:  # the arm got the same SIGINT and is saving; never kill it
                    stopped = True
                    print_fn(f'{arm}: interrupted, waiting for it to save', flush=True)
        record['run_dirs'] = sorted({os.path.dirname(p) for p in glob.glob(
            os.path.join(root, 'runs', arm, '**', 'flags.json'), recursive=True)})
        record['exit_code'] = code
        record['status'] = 'stopped' if stopped else ('done' if code == 0 else 'failed')
        save_session(path, session)
        print_fn(f'=== {arm}: {record["status"]} (exit {code})', flush=True)
        if record['status'] != 'done':
            return code or 1
    print_fn('=== all arms done', flush=True)
    return 0


def launch_runner(root):
    """Start `run_arms` detached from this terminal, in its own session. Returns its pid."""
    os.makedirs(os.path.join(root, 'logs'), exist_ok=True)
    log = open(os.path.join(root, 'logs', 'runner.log'), 'a')
    proc = subprocess.Popen(
        [sys.executable, '-m', 'deploy.spot_session', '--run', root],
        cwd=REPO, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
    )  # fmt: skip
    return proc.pid


def _ask_pose(prompt, client, input_fn, default):
    """Robot: the vision pose when the operator presses Enter. Sim: Enter, or 'x,y[,yaw]' in the grid."""
    if default is None:
        input_fn(prompt)
        s = client.get_state()
        return s['x'], s['y'], s['yaw']
    text = input_fn(f'{prompt} [sim: Enter for {default[:2]}, or type x,y]').strip()
    if not text:
        return default
    v = [float(t) for t in text.split(',')]
    return v[0], v[1], v[2] if len(v) > 2 else 0.0


def run_session(root, hostname, make_client, input_fn=input, print_fn=print, launch_fn=launch_runner, sim=False):
    """The guided flow. Returns the session dict it launched, or None when it launched nothing."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, 'session.json')
    session = None
    if os.path.exists(path):
        old = load_session(path)
        pid = old.get('runner_pid')
        if pid and _pid_alive(pid):
            print_fn(f'A runner (pid {pid}) is still driving Spot. Stop it first: kill -INT -- -{pid}')
            return None
        answer = ''
        while answer not in ('r', 'n'):
            answer = input_fn(f'A previous session exists in {root}. [r]esume / [n]ew: ').strip().lower()[:1]
        if answer == 'r':
            session = old
        else:
            old_dir = os.path.join(root, 'old_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.makedirs(old_dir)
            for name in os.listdir(root):
                if not name.startswith('old_'):
                    shutil.move(os.path.join(root, name), old_dir)
            print_fn(f'Previous session moved to {old_dir}')

    client = make_client()
    tag_id = None if session is None else session['tag_id']
    input_fn('Point Spot at the AprilTag and press Enter.')
    found = client.detect_fiducial(tag_id)
    while found is None:
        input_fn(NOT_DETECTED)
        found = client.detect_fiducial(tag_id)
    tag_id, tag_pose = found
    frame = RunFrame(tag_pose)
    print_fn(f'Tag {tag_id} found.')

    if session is None:
        end = frame.to_run(*_ask_pose('Walk Spot to the END point, then press Enter.', client, input_fn,
                                      SIM_END if sim else None))
        start = frame.to_run(*_ask_pose('Walk Spot back to the START (near the tag), then press Enter.', client,
                                        input_fn, SIM_START if sim else None))
        session = dict(
            tag_id=tag_id,
            end_tag=list(end),
            start_tag=list(start),
            created=datetime.datetime.now().isoformat(timespec='seconds'),
            hostname=hostname,
            arms=list(ARMS),
            runs={arm: dict(status='pending', run_dirs=[]) for arm in ARMS},
        )
        print_fn(f'Saved in the tag frame: end {end[:2]}, start {start[:2]}')
        if math.hypot(end[0] - start[0], end[1] - start[1]) < 1.0:  # twice the default goal_tol
            print_fn('WARNING: end and start are under 1 m apart, so one step can reach both goals.')
    else:
        print_fn(f'Resuming: end {session["end_tag"][:2]}, start {session["start_tag"][:2]} (tag frame)')
    session['tag_pose'] = list(tag_pose)  # vision frame on the robot, grid frame in the sim
    session['hostname'] = hostname
    save_session(path, session)

    input_fn('Release control on the tablet now; Spot will stand and start moving. Press Enter to start.')
    pid = launch_fn(root)
    print_fn(
        f'Running the arms in the background (runner pid {pid}).\n'
        f'  Logs:    {os.path.join(root, "logs")}/{{runner,sharsa,dual,phys,geo}}.log\n'
        f'  Status:  {path}\n'
        f'  Stop:    Ctrl+C here, or spot stop   (the running arm saves its buffer and weights first)\n'
        f'  Resume:  spot, then answer r'
    )
    return session


def main(sim=False):
    if sim:
        root = os.environ.get('SPOT_SESSION_ROOT', os.path.join(REPO, 'exp', 'spot_session_sim'))
        hostname, make_client = 'sim', SimSessionClient
    else:
        missing = [k for k in ('SPOT_IP', 'SPOT_USERNAME', 'SPOT_PASSWORD') if not os.environ.get(k)]
        if missing:
            sys.exit(f'Set {", ".join(missing)} in the environment before running the Spot session.')
        root = os.environ.get('SPOT_SESSION_ROOT', '/mnt/data/spot_session')
        hostname = os.environ['SPOT_IP']

        def make_client():
            from deploy.spot_client import SpotClient

            client = SpotClient(hostname, pose_source='vision')
            client.connect()  # no lease: the operator drives with the tablet during setup
            return client

    print('Spot session: face the AprilTag, mark the END, mark the START, then the four arms run on their own.')
    run_session(os.path.abspath(root), hostname, make_client, sim=sim)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--run']:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)  # survive the SSH session going away
        sys.exit(run_arms(sys.argv[2]))
