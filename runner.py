import argparse
import os
import subprocess
import tempfile
import pygit2

"""
Example:
python.exe $ENV(PWD)/runner.py $ENV(PWD)/torch_venv/bin/python.exe $ENV(HOME) master train.py --id master-weka --kernel-size 3 --output-dir /mnt/fast/nobackup/scratch4weeks/oh00320/denoising-pt/outputs/ --dataset-dir /mnt/fast/nobackup/users/oh00320/dataset/train/ --num-epochs 100 --patience 5 --min-delta 0.005

Everything that comes after [target_file] is the arguments for said file
Note that we only need to specify the name of [target_file], not the path, as it should exist only after this scripts clones the branch into /tmp/
"""

def parse_known_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", help="Executable used to call the script.")
    parser.add_argument("user_home_directory", help="Home directory containing .ssh/ folder")
    parser.add_argument("git_branch", help="Git branch of target file to be ran")
    parser.add_argument("target_file", help="Target file in cloned repository to run with remaining arguments")
    return parser.parse_known_args()

def create_temp_directory():
    try:
        temporary_directory = tempfile.TemporaryDirectory(prefix="denoising-pt")
        return temporary_directory
    except RuntimeError as error:
        print(f"Could not create a working directory: {error}")
        return None


def clone_repository(branch_name, ssh_public_key_path, ssh_private_key_path, path):
    try:
        ssh_keypair = pygit2.Keypair(username="git", 
                                     pubkey=ssh_public_key_path,
                                     privkey=ssh_private_key_path, 
                                     passphrase="")
        git_callbacks = pygit2.RemoteCallbacks(credentials=ssh_keypair)
        repository = pygit2.clone_repository("git@gitlab.surrey.ac.uk:oh00320/NeuralMCDenoiser.git", 
                                             path,
                                             checkout_branch=branch_name, 
                                             callbacks=git_callbacks)
        return repository.head.target
    except pygit2.GitError as error:
        print(f"Could not clone the repository: {error}")


def main():
    required_args, remainder_args = parse_known_cli_args()

    git_branch = required_args.git_branch

    if required_args.user_home_directory is not None:
        os.environ['HOME'] = required_args.user_home_directory
        print(f"HOME environment variable set to: {required_args.user_home_directory}")
    try:
        user_home_directory = os.environ['HOME']
    except:
        print("Could not retrieve user home directory. Aborting")
        return

    working_directory = create_temp_directory()
    if not working_directory:
        return
    print(f"Working directory: {working_directory.name}")

    ssh_key_id = "id_rsa"
    ssh_key_path_prefix = os.path.join(user_home_directory, ".ssh")
    ssh_public_key_path = os.path.join(ssh_key_path_prefix, ssh_key_id + ".pub")
    ssh_private_key_path = os.path.join(ssh_key_path_prefix, ssh_key_id)
    commit_id = clone_repository(git_branch, ssh_public_key_path, ssh_private_key_path, working_directory.name)
    if commit_id is None:
        return
    print(f"Commit ID: {commit_id}")

    start_point = os.path.join(working_directory.name, required_args.target_file)
    subprocess_args = [required_args.executable, start_point] + remainder_args
    try:
        subprocess.call(args=subprocess_args)
    except:
        print("Could not run the package")

if __name__ == "__main__":
    main()
