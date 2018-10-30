import os, random, time, subprocess

pages=['/channels/staffpicks/252069857?autoplay=1', '/channels/staffpicks/252220553?autoplay=1', '/channels/staffpicks/251713531?autoplay=1', '/channels/staffpicks/251632428?autoplay=1', '/channels/staffpicks/251646742?autoplay=1', '/channels/staffpicks/251853214?autoplay=1', '/channels/staffpicks/249754664?autoplay=1', '/channels/staffpicks/180891891?autoplay=1', '/channels/staffpicks/252417024?autoplay=1', '/channels/staffpicks/217342747?autoplay=1', '/channels/staffpicks/252441577?autoplay=1', '/channels/staffpicks/251397575?autoplay=1', '/channels/staffpicks/250356669?autoplay=1', '/channels/staffpicks/249449543?autoplay=1', '/channels/staffpicks/248505397?autoplay=1', '/channels/staffpicks/245840069?autoplay=1', '/channels/staffpicks/240839522?autoplay=1', '/channels/staffpicks/244683457?autoplay=1', '/channels/staffpicks/247075867?autoplay=1', '/channels/staffpicks/246779079?autoplay=1', '/channels/staffpicks/244729509?autoplay=1', '/channels/staffpicks/244405542?autoplay=1', '/channels/staffpicks/248505397?autoplay=1', '/channels/staffpicks/244969966?autoplay=1', '/channels/staffpicks/244882764?autoplay=1', '/channels/staffpicks/202284096?autoplay=1', '/channels/staffpicks/244405542?autoplay=1', '/channels/staffpicks/154583964?autoplay=1', '/channels/staffpicks/212731897?autoplay=1', '/channels/staffpicks/196300116?autoplay=1']
domain='https://vimeo.com'
killChrome=1
while True:
	if killChrome:
		os.system("taskkill /IM chrome.exe")
		time.sleep(1)
	url= domain+ random.choice(pages)
	#os.system("start "+ url)
	p = subprocess.Popen(["C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", "--incognito" , url ])
	time.sleep(60)