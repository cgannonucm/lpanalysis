#!/usr/bin/env python
import symlib
import os 

def main():
    #You will need your own symphony credential, put them in symphony credentials.txt
    #with login as first line, password second line
    with open("symphony_credentials.txt") as f:
        lines = f.readlines()
        user = lines[0].strip()
        password = lines[1].strip()
    
    data_dir = "data/symphony"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # The dataset you want to download.
    target = "halos"
    
    # Download all the host halos in the Milky Way-mass suite.
    #symlib.download_files(user, password, "SymphonyGroup", None,
    #        data_dir, target=target)

    symlib.download_files(user, password, "SymphonyMilkyWay", None,
            data_dir, target=target)

if __name__ == "__main__":
    main()
