def prob_local_ip():
	import socket
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.connect(("8.8.8.8", 80))
	ip = s.getsockname()[0]
	print('using local ip address',ip)
	s.close()
	ip = ip.split('.')
	return '%s%s%s%s' %(chr(int(ip[0])), chr(int(ip[1])), chr(int(ip[2])), chr(int(ip[3])))

def get_statistics(key,item):
	result = []
	return False