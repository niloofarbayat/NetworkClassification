#***********************************************************************************
# Valid domain characters
#***********************************************************************************
def isDomainChar(c):
	if c.isalpha() or c.isdigit() or c==b'-' or c==b'_' or c==b'.' or c==b'/':
		return True
	return False

#***********************************************************************************
# Reverses the format dst_ip+dport+src_ip+sport to src_ip+sport+dst_ip+dport
#***********************************************************************************
def conn_id_reverse(id):
	return id[4:8] + id[0:4] + id[10:12] + id[8:10]

#***********************************************************************************
# Checks for remote vs local
#***********************************************************************************
def isLocalAddress(s):
	return s.startswith(b'\x0a') or s.startswith(b'\xc0\xa8') #10.x.x.x, 192.168.x.x

#***********************************************************************************
# Gets connection id
# unify both direct (src_ip+sport+dst_ip+dport) and reverse (dst_ip+dport+src_ip+sport)
# network traffic
#***********************************************************************************
def unify_conn_id(conn_id):
	l = isLocalAddress(conn_id[0:4])
	r = isLocalAddress(conn_id[4:8])
	if (l==r):
		if conn_id[0:4] < conn_id[4:8]:
			return conn_id, 1
		else:
			return conn_id_reverse(conn_id), 0
	#else l!=r
	if not l:
		return conn_id_reverse(conn_id), 0
	return conn_id, 1

#***********************************************************************************
# Gets ip position
#***********************************************************************************
def get_ip_pos(data):
	ip_pos = 14
	if data[12:14] == b'\x81\x00':
		ip_pos = 18
	if data[ip_pos-2:ip_pos-1]==b'\x08' and data[ip_pos-1:ip_pos]==b'\x00':
		return ip_pos
	else:
		return -1 #not ip protocol
