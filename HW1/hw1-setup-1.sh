FILE_ID=1V_BxpqGIEO2g33czjs7wDcWa4iTkSTl3
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILE_ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILE_ID" -O ECE228-Homework-1.tar.gz && rm -rf /tmp/cookies.txt
tar -xvf ECE228-Homework-1.tar.gz