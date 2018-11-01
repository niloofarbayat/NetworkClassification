def isDomainChar(c):
	if c.isalpha() or c.isdigit() or c=='-' or c=='_' or c=='.' or c=='/':
		return True
	return False

def conn_id_reverse(id):
	return '%s%s%s%s%s' % (id[4:8], id[0:4], id[10:12], id[8:10], id[12])

def isLocalAddress(s, local_ip):
	if s.startswith('\x0a') or s.startswith('\xc0\xa8'): #10.x.x.x, 192.168.x.x
		return True
	return s.startswith(local_ip)

def unify_conn_id(conn_id, local_ip):
	l = isLocalAddress(conn_id[0:4], local_ip)
	r = isLocalAddress(conn_id[4:8], local_ip)
	if (l==r):
		if conn_id[0:4] < conn_id[4:8] :
			return conn_id, 1
		else:
			return conn_id_reverse(conn_id), 0
	#else l!=r
	if not l:
		return conn_id_reverse(conn_id), 0
	return conn_id, 1

def prob_local_ip():
	import socket
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.connect(("8.8.8.8", 80))
	ip = s.getsockname()[0]
	print('using local ip address',ip)
	s.close()
	ip = ip.split('.')
	return '%s%s%s%s' %(chr(int(ip[0])), chr(int(ip[1])), chr(int(ip[2])), chr(int(ip[3])))

def get_ip_pos(data):
	ip_pos = 14
	if data[12:14] == '\x81\x00':
		ip_pos = 18
	if data[ip_pos-2]=='\x08' and data[ip_pos-1]=='\x00':
		return ip_pos
	else:
		return -1 #not ip protocol