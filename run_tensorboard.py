import os 
import subprocess


log_dir = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop/tensorboard_Data')


port = 6111 ## random.randint(6000, 7000)
subprocess.Popen(f"tensorboard --logdir={log_dir} --port={port} --reload_multifile=true", shell=True)
