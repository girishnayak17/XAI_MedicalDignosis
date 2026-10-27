#!/usr/bin/env python3

import urllib.request
import os
import time

# URLs for the files
links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
    'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
    'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
    'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
    'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
    'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
    'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
    'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
]


def download_file(url, filename):

    # Check if file already exists
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        file_size = os.path.getsize(filename) / (1024 * 1024)

        print(f"\n[SKIP] {filename} already exists")
        print(f"       Existing file size: {file_size:.2f} MB")
        return

    print(f"\nDownloading: {filename}")

    try:
        response = urllib.request.urlopen(url)

        total_size = int(response.headers.get('Content-Length', 0))

        downloaded = 0
        start_time = time.time()

        # Download in 1 MB chunks
        chunk_size = 1024 * 1024

        with open(filename, 'wb') as file:

            while True:

                chunk = response.read(chunk_size)

                if not chunk:
                    break

                file.write(chunk)

                downloaded += len(chunk)

                # Calculate percentage
                if total_size > 0:
                    percent = downloaded * 100 / total_size
                else:
                    percent = 0

                # Calculate speed
                elapsed = time.time() - start_time

                if elapsed > 0:
                    speed = downloaded / elapsed
                else:
                    speed = 0

                # Convert speed to MB/s
                speed_mb = speed / (1024 * 1024)

                # Convert downloaded size to MB
                downloaded_mb = downloaded / (1024 * 1024)

                # Convert total size to MB
                total_mb = total_size / (1024 * 1024)

                # Display progress
                print(
                    f"\rProgress: {percent:6.2f}% "
                    f"| {downloaded_mb:8.2f}/{total_mb:8.2f} MB "
                    f"| Speed: {speed_mb:6.2f} MB/s",
                    end="",
                    flush=True
                )

        print("\nDownload completed!")

    except Exception as e:
        print(f"\nError downloading {filename}: {e}")


# Download all files
for idx, link in enumerate(links):

    filename = f'images_{idx + 1:02d}.tar.gz'

    download_file(link, filename)


print("\n===================================")
print("All downloads completed!")
print("===================================")