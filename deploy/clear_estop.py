"""Clear keepalive policies left behind by a run that died. Operator recovery tool.

    python deploy/clear_estop.py --robot_hostname=$SPOT_IP

Registering an e-stop endpoint also registers a keepalive policy that cuts motor power when
check-ins lapse. A crashed run leaves that policy behind, the timer fires, and the robot then
refuses to power on with `KeepaliveMotorsOffError` while the e-stop itself still reports
ESTOP_LEVEL_NONE, which makes the cause hard to see.

This is deliberately a separate command rather than something `acquire()` does on every start: the
filter cannot tell a dead session's policy from a live one belonging to another operator, so the
decision to remove a motor-off protection stays with the person at the robot. It prints every
policy with its age before touching anything, and `--dry_run` stops there.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl import app, flags
from bosdyn.client import create_standard_sdk
from bosdyn.client.keepalive import KeepaliveClient

FLAGS = flags.FLAGS

flags.DEFINE_string('robot_hostname', None, 'Spot address.')
flags.DEFINE_bool('dry_run', False, 'List policies and exit without removing any.')
flags.DEFINE_float('stale_after', 15.0, 'Seconds without a check-in before a policy counts stale.')


def main(_):
    assert FLAGS.robot_hostname is not None, 'pass --robot_hostname'

    sdk = create_standard_sdk('clear-estop')
    robot = sdk.create_robot(FLAGS.robot_hostname)
    robot.authenticate(os.environ['SPOT_USERNAME'], os.environ['SPOT_PASSWORD'])
    robot.time_sync.wait_for_sync()
    client = robot.ensure_client(KeepaliveClient.default_service_name)

    now = time.time()
    removed, kept = 0, 0
    for live in client.get_status().status:
        cuts_power = any(a.HasField('controlled_motors_off') for a in live.policy.actions)
        age = now - live.last_checkin.seconds
        label = live.policy.name or '(unnamed)'
        kind = 'motors-off' if cuts_power else 'other'
        stale = age > FLAGS.stale_after

        if cuts_power and stale:
            print(f'REMOVE  {label:40s} {kind:11s} last check-in {age:.0f}s ago')
            if not FLAGS.dry_run:
                client.modify_policy(policy_ids_to_remove=[live.policy_id])
            removed += 1
        else:
            why = 'still checking in' if cuts_power else 'does not cut motor power'
            print(f'keep    {label:40s} {kind:11s} last check-in {age:.0f}s ago, {why}')
            kept += 1

    verb = 'would remove' if FLAGS.dry_run else 'removed'
    print(f'\n{verb} {removed}, kept {kept}')
    if removed and not FLAGS.dry_run:
        print('motor power should now be grantable; rerun the task command')


if __name__ == '__main__':
    app.run(main)
