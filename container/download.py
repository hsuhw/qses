import urllib.request
import sys


def get_one_file(url,file_name,size_KB):
    size_KB=int(size_KB)
    req=urllib.request.urlopen(url)
    filename=file_name
    _file=open(filename,'wb')
    _file.write(req.read())
    get_bytes=_file.seek(0,2)
    if get_bytes>=size_KB*1000:
        print('Download finished')
    else:
        print("Download unfinished,retry...")
        get_one_file(url,file_name,size_KB)

get_one_file(sys.argv[1],sys.argv[2],sys.argv[3])
