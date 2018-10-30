import re
with open('page.html','r') as f:
	s = f.read()
# /channels/staffpicks/252069857?autoplay=1
d={}
for link in re.findall('/watch/\d+\?',s):
	d[link[:-1]]=0
links=[]
for link in d:
	links.append(link)
print links