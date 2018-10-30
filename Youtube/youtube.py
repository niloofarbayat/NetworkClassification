import os,time,sys,subprocess,random

with open('linklist.txt','r') as f:
	linklist = f.readlines()
print linklist
ind=-1
while 1:
	os.system("killall -9 \"Google Chrome\"")
	ind+=1
	p = subprocess.Popen(["/usr/bin/open -a /Applications/Google\ Chrome.app " +  random.choice(linklist).strip() ], shell=True)
	time.sleep(10)
	#p.kill()
