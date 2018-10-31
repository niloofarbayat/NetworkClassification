import subprocess, struct, time, select, threading, os, sys, traceback, itertools,math, collections
from ctypes import *
from pytcpdump_utils import *

#Help Info:
#python pytcpdump 
interface = "br-lan"
snaplen = '1600'
log_https_hello = '/tmp/https_hello.pcap'
fromfile_https_hello = False
snaplen_hdr = '80'
log_pkt_hdr = '/tmp/pkt_hdr.pcap' #/dev/null
fromfile_pkt_hdr = False
LRU_size = 100000 #TODO: This could be a problem
lru = None
local_ip = prob_local_ip()
threading_https_hello = None
threading_pkt_hdr = None
fromfile_path = None

def pytcpdump():
	global lru, threading_https_hello, threading_pkt_hdr
	lru = LRUCache(LRU_size)
	if fromfile_path:
		process_file(fromfile_path)
	else:
		threading_https_hello = threading.Thread(target=thr_https_hello)
		threading_https_hello.start()

		threading_pkt_hdr = threading.Thread(target=thr_pkt_hdr)
		threading_pkt_hdr.start()
		
class pcap_hdr_s(Structure):
	_fields_ = [('magic', c_uint32),
				('v1', c_uint16),
				('v2', c_uint16),
				('zone', c_uint32),
				('sigfigs', c_uint32),
				('snaplen', c_uint32),
				('network', c_uint32)]
class pcaprec_hdr_s(Structure):
	_fields_ = [('sec', c_uint32),
				('usec', c_uint32),
				('len', c_uint32),
				('olen', c_uint32)]
def isDomainChar(c):
	if c.isalpha() or c.isdigit() or c=='-' or c=='_' or c=='.' or c=='/':
		return True
	return False
def unify_conn_id(conn_id):
	l = isLocalAddress(conn_id[0:4])
	r = isLocalAddress(conn_id[4:8])
	if (l==r):
		if conn_id[0:4] < conn_id[4:8] :
			return conn_id, 1
		else:
			return conn_id_reverse(conn_id), 0
	#else l!=r
	if not l:
		return conn_id_reverse(conn_id), 0
	return conn_id, 1
def isLocalAddress(s):
	if s.startswith('\x0a') or s.startswith('\xc0\xa8'): #10.x.x.x, 192.168.x.x
		return True
	return s.startswith( local_ip )


class LRUCache:
	def __init__(self, capacity):
		self.capacity = capacity
		self.lock = threading.RLock()
		self.cache = collections.OrderedDict()
	#index help:
	#0:str:hostname, 
	#1:[int,int]:accumulated bytes[remote->local, local->remote], 
	#2:list[[int],[int]]:arrival seconds, 
	#3:list[[int],[int]]:arrival micro-seconds, 
	#4:list[[int],[int]]:packet size
	#5:list[[int],[int]]:tcp payload size
	def set_hostname(self, key, hostname):
		with self.lock:
			item = self.pop(key)
			item[0] = hostname
			self.cache[key] = item
			
	def update(self, key, fromLocal, pcaprec_hdr, payload_len):
		with self.lock:
			item = self.pop(key)
			item[1][fromLocal]+=pcaprec_hdr.olen
			item[2][fromLocal].append(pcaprec_hdr.sec)
			item[3][fromLocal].append(pcaprec_hdr.usec)
			item[4][fromLocal].append(pcaprec_hdr.olen)
			item[5][fromLocal].append(payload_len)
			self.cache[key] = item
			
	def pop(self, key):
		with self.lock:
			try:
				return self.cache.pop(key)
			except KeyError:
				return ['unknown',[0,0],[[],[]],[[],[]],[[],[]],[[],[]]]

	def get(self, key): 
		with self.lock:
			try:
				value = self.cache.pop(key)
				self.cache[key] = value
				return value
			except KeyError:
				return -1

	def set(self, key, value):
		with self.lock:
			try:
				self.cache.pop(key)
			except KeyError:
				if len(self.cache) >= self.capacity:
					self.cache.popitem(last=False)
			self.cache[key] = value
			
	def print_latest(self, count):
		with self.lock:
			conn_index = 0
			for key in reversed(self.cache):
				item = self.cache[key]
				print conn_id_to_str(key), item[0:2], '#pkts', len(item[4][0]),len(item[4][1])
				conn_index+=1
				if conn_index==count:
					return
	def snapshot(self, count):
		ret=[]
		with self.lock:
			for key in reversed(self.cache):
				ret.append((key, self.cache[key]))
				count-=1
				if count==0:
					break
			return ret

def sni_pos(data):
	pos=0
	while 1:
		pos=data.find('.',pos)
		if pos<0:
			return -1,-1
		pos1=pos
		while isDomainChar(data[pos1-1]):
			pos1-=1
		pos2=pos
		while 1:
			pos2+=1
			if pos2==len(data):
				break
			if not isDomainChar(data[pos2]):
				break
		if pos2-pos1>=5:
			if ord(data[pos1-1])==0: #data[pos1-2:pos1] should be the len
				pos1+=1
			if (ord(data[pos1-2])<<8)+ord(data[pos1-1]) == pos2-pos1:
				#print pkt_hdr.sec, pkt_hdr.usec, data[pos1:pos2], the_time
				return pos1, pos2
		pos=pos2
	
def conn_id_to_str(id):
	transport='TCP'
	if id.endswith('u'):
		transport='UDP'
	return '%d.%d.%d.%d:%d->%d.%d.%d.%d:%d %s' %(ord(id[0]), ord(id[1]), ord(id[2]), ord(id[3]), (ord(id[8])<<8)+ord(id[9]), ord(id[4]), ord(id[5]), ord(id[6]), ord(id[7]), (ord(id[10])<<8)+ord(id[11]), transport)

def conn_id_reverse(id):
	return '%s%s%s%s%s' % (id[4:8], id[0:4], id[10:12], id[8:10], id[12])

def get_ip_pos(data):
	ip_pos = 14
	if data[12:14] == '\x81\x00':
		ip_pos = 18
	if data[ip_pos-2]=='\x08' and data[ip_pos-1]=='\x00':
		return ip_pos
	else:
		return -1 #not ip protocol
	
def process_file(filename):
	global lru
	if not lru:
		lru = LRUCache(LRU_size)
	pcap_hdr = pcap_hdr_s()
	pkt_hdr = pcaprec_hdr_s()
	pkt_hdr_size = sizeof(pkt_hdr)
	with open(filename,'rb') as f:
		f.readinto(pcap_hdr)
		while 1:
			if f.readinto(pkt_hdr)!=pkt_hdr_size:
				break
			#print pkt_hdr.len, pkt_hdr.olen, pkt_hdr.sec, pkt_hdr.usec
			data = f.read(pkt_hdr.len)

			ip_pos = get_ip_pos(data) # IP layer
			if ip_pos<0:
				continue
			tcp_pos = ip_pos + ((ord(data[ip_pos]) & 0x0f)<<2) #mostly equals ip_pos+20 # TCP layer
			ip_proto = ord(data[ip_pos+9])
			if  ip_proto == 0x06: #tcp
				conn_id = data[ip_pos+ 12: ip_pos+20]+data[tcp_pos : tcp_pos+4] + 't'
			elif ip_proto == 0x17: #udp
				continue
				conn_id = data[ip_pos+ 12: ip_pos+20]+data[tcp_pos : tcp_pos+4] + 'u'
			else:
				continue
			conn_id , fromLocal = unify_conn_id(conn_id)
			tcp_load_pos = tcp_pos + (( ord(data[tcp_pos+12]) & 0xf0 )>>2)
			lru.update(conn_id, fromLocal, pkt_hdr, pkt_hdr.olen-tcp_load_pos)
			if (tcp_load_pos+5<len(data)) and ( ord(data[tcp_load_pos]) == 0x16) and (ord(data[tcp_load_pos+5]) == 0x01):
				#print 'https hello packet'
				pos1, pos2 = sni_pos(data)
				if pos1>0:
					lru.set_hostname(conn_id, data[pos1:pos2])

def thr_https_hello():
	#sudo tcpdump -i eno1 -s 1500 -n "(tcp[((tcp[12:1] & 0xf0) >> 2)+5:1] = 0x01) and (tcp[((tcp[12:1] & 0xf0) >> 2):1] = 0x16)" -w /tmp/sni.pcap
	if fromfile_https_hello:
		p = subprocess.Popen(('tcpdump', '-U', '-s', snaplen, '-w', '-', '(tcp[((tcp[12:1] & 0xf0) >> 2)+5:1] = 0x01) and (tcp[((tcp[12:1] & 0xf0) >> 2):1] = 0x16)','-r',fromfile_https_hello), stdout=subprocess.PIPE)
	else:
		if log_https_hello:
			p = subprocess.Popen(('tcpdump -B 2048 -U -i %s -s %s -w - "(tcp[((tcp[12:1] & 0xf0) >> 2)+5:1] = 0x01) and (tcp[((tcp[12:1] & 0xf0) >> 2):1] = 0x16)" |tee %s' % (interface, snaplen, log_https_hello)), stdout=subprocess.PIPE, shell=True)
		else:
			p = subprocess.Popen(('tcpdump', '-U', '-i', interface, '-s', snaplen, '-w', '-', '(tcp[((tcp[12:1] & 0xf0) >> 2)+5:1] = 0x01) and (tcp[((tcp[12:1] & 0xf0) >> 2):1] = 0x16)'), stdout=subprocess.PIPE)
	try:
		pcap_hdr = pcap_hdr_s()
		#print "pcap_hdr read, len =", 
		p.stdout.readinto(pcap_hdr)
		#print "snaplen", pcap_hdr.snaplen,'network', pcap_hdr.network
		pkt_hdr = pcaprec_hdr_s()
		pkt_hdr_size = sizeof(pkt_hdr)
		while 1:
			#prev_time = time.time()
			#while 1:
			#	readable, writable, exceptional = select.select([p.stdout],[],[p.stdout],0.01)
			#	if readable:
			#		break
			#	print 'waiting',time.time()
			if p.stdout.readinto(pkt_hdr)!=pkt_hdr_size:
				break
			#the_time = time.time()
			#print 'pkt read', pkt_hdr.sec, pkt_hdr.usec, pkt_hdr.len, pkt_hdr.olen
			data = p.stdout.read(pkt_hdr.len)
			#print "mac1", ':'.join(x.encode('hex') for x in data[0:6]) #dest MAC
			#print "mac2", ':'.join(x.encode('hex') for x in data[6:12]) #src MAC
			#print 'ip len', ':'.join(x.encode('hex') for x in data[16:18]), struct.unpack(">H", data[16:18])[0]
			#tcp_pos = 
			#sni_len=
			#print 'ip data', ':'.join(x.encode('hex') for x in data[18:22]), struct.unpack(">H", data[20:22])[0]
			pos1, pos2 = sni_pos(data)
			if pos1>0:
				ip_pos = get_ip_pos(data)
				tcp_pos = ip_pos + ((ord(data[ip_pos]) & 0x0f)<<2) #mostly equals ip_pos+20
				conn_id = data[ip_pos+ 12: ip_pos+20]+data[tcp_pos : tcp_pos+4] + 't'
				conn_id , _ = unify_conn_id(conn_id)
				lru.set_hostname(conn_id, data[pos1:pos2])								
	except Exception, e:
		traceback.print_exc()
		print 'tcpdump https hello error'
	p.kill()

def thr_pkt_hdr():
	if fromfile_pkt_hdr:
		p = subprocess.Popen(('tcpdump', '-s', snaplen_hdr, '-w', '-', "tcp", '-r', fromfile_pkt_hdr), stdout=subprocess.PIPE)
	else:
		if log_pkt_hdr:
			p = subprocess.Popen(('tcpdump -B 2048 -U --immediate-mode -n -i %s -s %s -w - |tee %s' % (interface, snaplen_hdr, log_pkt_hdr)), stdout=subprocess.PIPE, shell=True)
			#p = subprocess.Popen(('tcpdump -B 4096 -U -i %s -w - ' % (interface,)), stdout=subprocess.PIPE, shell=True)
		else:
			p = subprocess.Popen(('tcpdump', '-U', '-i', interface, '-s', snaplen_hdr, '-w', '-', "tcp"), stdout=subprocess.PIPE)
	try:
		pcap_hdr = pcap_hdr_s()
		p.stdout.readinto(pcap_hdr)
		pkt_hdr = pcaprec_hdr_s()
		pkt_hdr_size = sizeof(pkt_hdr)
		while 1:
			if p.stdout.readinto(pkt_hdr)!=pkt_hdr_size:
				break
			data = p.stdout.read(pkt_hdr.len)
			ip_pos = get_ip_pos(data)
			if ip_pos<0:
				continue
			tcp_pos = ip_pos + ((ord(data[ip_pos]) & 0x0f)<<2) #mostly equals ip_pos+20
			ip_proto = ord(data[ip_pos+9])
			if  ip_proto == 0x06: #tcp
				conn_id = data[ip_pos+ 12: ip_pos+20]+data[tcp_pos : tcp_pos+4] + 't'
			elif ip_proto == 0x17: #udp
				continue
				conn_id = data[ip_pos+ 12: ip_pos+20]+data[tcp_pos : tcp_pos+4] + 'u'
			else:
				continue
			conn_id , fromLocal = unify_conn_id(conn_id)
			lru.update(conn_id, pkt_hdr, fromLocal)
			# print conn_id_to_str(conn_id)
	except Exception, e:
		traceback.print_exc()
		print 'tcpdump thread error'
	p.kill()

def thr_print_latest():
	while 1:
		if lru.cache:
			os.system('clear')
		lru.print_latest(30)
		time.sleep(2)
def stat_head():
	return "class,id,sni,size25,size50,size75,sizeMax,sizeAvg,sizeVar,iat25,iat50,iat75,iatMax,iatAvg,iatVar,Csize25,Csize50,Csize75,CsizeMax,CsizeAvg,CsizeVar,Ciat25,Ciat50,Ciat75,CiatMax,CiatAvg,CiatVar\n"
def stat_calc(x, limit):
	try:
		l=len(x)
		if l==0:
			return [str(a) for a in [0,0,0,0,0,0]]
		if l==1:
			return [str(a) for a in [x[0], x[0], x[0], x[0], x[0], 0]]
		if len(x)>limit:
			x=x[:limit]
			l=len(x)
		x = sorted(x)
		avg = float(sum(x)) / l
		var = sum([(xi - avg) ** 2 for xi in x]) / l
		return [str(a) for a in [x[int(round((l-1)/4.0))],x[int(round((l-1)/2.0))],x[int(round((l-1)*3/4.0))],max(x),avg,var] ]
	except Exception as e:
		traceback.print_exc()
def stat_prepare_iat(sec,  usec):
	l = len(sec)
	t = [sec[i]+usec[i]*1e-6 for i in xrange(l)]
	return [t[i+1]-t[i] for i in xrange(l-1)]
def stat_create(data,filename, nlimit, googlevideo):
	with open(filename,'wb') as f:
		f.write(stat_head())
		for id in data:
			item=data[id]
			if googlevideo == (item[0].endswith('googlevideo.com') and item[0].find('-')>=0 ):
				line=['video' if googlevideo else 'web',conn_id_to_str(id),item[0]]
				#remote->local
				line+=stat_calc(item[5][0], nlimit)
				iat = stat_prepare_iat(item[2][0],item[3][0])
				line+=stat_calc(iat, nlimit-1)
				#local->remote
				line+=stat_calc(item[5][1], nlimit)
				iat = stat_prepare_iat(item[2][1],item[3][1])
				line+=stat_calc(iat, nlimit-1)
				line= ','.join(line)
				f.write(line)
				f.write('\n')
				'''
				print conn_id_to_str(id)
				print 'received sizes'
				dir=0
				for i in range(len(item[4][dir])):
					print '  ',item[4][dir][i],item[5][dir][i],
					if i>0:
						print "%.6f" %( (item[2][dir][i]+item[3][dir][i]*1e-6)-(item[2][dir][i-1] +item[3][dir][i-1]*1e-6), ),
					print ''
				print 'send sizes'
				dir =1
				dir=0
				for i in range(len(item[4][dir])):
					print '  ',item[4][dir][i],item[5][dir][i]
					if i>0:
						print "%.6f" %( (item[2][dir][i]+item[3][dir][i]*1e-6)-(item[2][dir][i-1] +item[3][dir][i-1]*1e-6), ),
					print ''
				'''
#Main
if __name__ == "__main__":
	for i in range(len(sys.argv)):
		if sys.argv[i]=='--https_file':
			fromfile_https_hello = sys.argv[i+1]
		elif sys.argv[i]=='--pkt_hdr_file':
			fromfile_pkt_hdr = sys.argv[i+1]
		elif sys.argv[i]=='--fromfile':
			fromfile_path = sys.argv[i+1]
		elif sys.argv[i]=='--local_ip':
			local_ip = sys.argv[i+1]
	pytcpdump()


	if (not fromfile_pkt_hdr) and (not fromfile_path):
		threading_print_latest = threading.Thread(target=thr_print_latest)
		threading_print_latest.daemon=True
		threading_print_latest.start()

	if threading_https_hello:
		threading_https_hello.join()
	if threading_pkt_hdr:
		threading_pkt_hdr.join()
	print "data read finished, keep",len(lru.cache),'records'
	#lru.print_latest(30)
	stat_create(lru.cache,'browse_limit2.csv', nlimit=2, googlevideo=False)
#for row in iter(p.stdout.readline, b''):
#	print row.rstrip()   # process here


