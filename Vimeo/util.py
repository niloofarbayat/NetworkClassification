import re
with open('html.txt','r') as f:
	s = f.read()
# /channels/staffpicks/252069857?autoplay=1
print re.findall('/channels/staffpicks/\d+\?autoplay=1',s)