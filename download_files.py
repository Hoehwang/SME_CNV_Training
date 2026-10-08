import os
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import paramiko

# SSH Proxy Gateway (Jump Host) Information
PROXY_HOSTNAME = "211.175.149.57"
PROXY_USERNAME = "genomom_lims"
PROXY_PORT = 3030

# Target SSH Connection Information
HOSTNAME = "211.175.149.59"
USERNAME = "genomom_lims"
PORT = 3030
PASSWORD = "DifFkcK9771%@"  # Both proxy and target share this password

# Source remote files
REMOTE_FILES = [
    "/BiO/SME_Synthetic_CNV/Thermo_Synthetic_Test/CNV_Synthetic_del/CNV_del_female_synthetic.csv",
    "/BiO/SME_Synthetic_CNV/Thermo_Synthetic_Test/CNV_Synthetic_del/CNV_del_male_synthetic.csv",
    "/BiO/SME_Synthetic_CNV/Thermo_Synthetic_Test/CNV_Synthetic_duple/CNV_duple_female_synthetic.csv",
    "/BiO/SME_Synthetic_CNV/Thermo_Synthetic_Test/CNV_Synthetic_duple/CNV_duple_male_synthetic.csv",
    "/BiO/SME_Synthetic_CNV/Thermo_Synthetic_Test/CNV_Synthetic_normal/CNV_normal_female_synthetic.csv",
    "/BiO/SME_Synthetic_CNV/Thermo_Synthetic_Test/CNV_Synthetic_normal/CNV_normal_male_synthetic.csv",
]

# Destination directory (current workspace)
LOCAL_DIR = "./Synthetic_Test"

def download_file_via_sftp(remote_path, local_path, position):
    file_name = os.path.basename(remote_path)
    pbar = None
    last_update = time.time()

    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):
        proxy_client = None
        target_transport = None

        def callback(transferred, total):
            nonlocal pbar, last_update
            if pbar is None:
                pbar = tqdm(
                    total=total,
                    initial=transferred,
                    unit="B",
                    unit_scale=True,
                    desc=file_name,
                    position=position,
                    leave=True
                )
            else:
                pbar.n = transferred
            
            current_time = time.time()
            # 5초가 경과했거나, 다운로드가 완료(100%)되었을 때만 화면 갱신
            if current_time - last_update >= 5.0 or transferred == total:
                pbar.refresh()
                last_update = current_time

        try:
            if attempt > 0:
                # Add random sleep before retry to stagger reconnection attempts
                time.sleep(random.uniform(2.0, 5.0))

            # 1. Connecting to Proxy Gateway
            proxy_client = paramiko.SSHClient()
            proxy_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            proxy_client.connect(
                hostname=PROXY_HOSTNAME,
                port=PROXY_PORT,
                username=PROXY_USERNAME,
                password=PASSWORD,
                timeout=15
            )
            
            # 2. Opening direct-tcpip channel
            proxy_transport = proxy_client.get_transport()
            dest_channel = proxy_transport.open_channel(
                "direct-tcpip",
                (HOSTNAME, PORT),
                ("127.0.0.1", 0),
                timeout=15
            )
            
            # 3. Connecting to Target Server
            target_transport = paramiko.Transport(dest_channel)
            target_transport.connect(username=USERNAME, password=PASSWORD)
            
            # Open an SFTP session channel on the target transport
            sftp = paramiko.SFTPClient.from_transport(target_transport)

            # Check sizes to determine if we need to resume
            remote_stat = sftp.stat(remote_path)
            remote_size = remote_stat.st_size
            
            local_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0

            if local_size == remote_size:
                if pbar is None:
                    print(f"\n[{file_name}] Already fully downloaded. Skipping.")
                sftp.close()
                break  # Success
                
            if local_size > remote_size:
                local_size = 0  # Re-download if local file is inexplicably larger
                
            mode = 'ab' if local_size > 0 else 'wb'

            with sftp.open(remote_path, 'rb') as rf:
                if local_size > 0:
                    rf.seek(local_size)
                
                with open(local_path, mode) as lf:
                    transferred = local_size
                    callback(transferred, remote_size)

                    # 1MB chunk size
                    chunk_size = 1024 * 1024
                    while True:
                        data = rf.read(chunk_size)
                        if not data:
                            break
                        lf.write(data)
                        transferred += len(data)
                        callback(transferred, remote_size)

            sftp.close()
            
            if pbar is not None:
                pbar.close()
            print(f"\n[{file_name}] Download completed successfully via SFTP!")
            break  # Success
            
        except Exception as e:
            if pbar is not None:
                pbar.close()
                pbar = None  # Reset pbar so it can be recreated on the next attempt
            print(f"\n[{file_name}] Error downloading (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES - 1:
                print(f"\n[{file_name}] Max retries reached. Download failed.")
        finally:
            if target_transport:
                try: target_transport.close()
                except: pass
            if proxy_client:
                try: proxy_client.close()
                except: pass


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    print("Starting concurrent downloads, each using an independent SSH connection...")
    
    with ThreadPoolExecutor(max_workers=len(REMOTE_FILES)) as executor:
        futures = []
        for idx, remote_path in enumerate(REMOTE_FILES):
            file_name = os.path.basename(remote_path)
            local_path = os.path.join(LOCAL_DIR, file_name)
            # Add a slight staggered delay to prevent multiple threads from hitting the SSH banner at the exact same millisecond
            time.sleep(1.0)
            futures.append(executor.submit(download_file_via_sftp, remote_path, local_path, idx))
        
        # Wait for all downloads to finish
        for future in futures:
            try:
                future.result()
            except Exception as e:
                pass

if __name__ == "__main__":
    main()