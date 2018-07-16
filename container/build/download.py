import sys
import urllib.request

def download(url, result_name, expected_size_kb):
    expected_size_kb = int(expected_size_kb)
    req = urllib.request.urlopen(url)
    file = open(result_name, 'wb')
    file.write(req.read())
    result_size = file.seek(0, 2)
    if result_size >= expected_size_kb * 1000:
        print('`%s` download succeeded.' % result_name)
    else:
        print('`%s` download failed, retrying...' % result_name)
        download(url, result_name, expected_size_kb)

def main(argv):
    download(argv[1], argv[2], argv[3])

if __name__ == '__main__':
    main(sys.argv)
