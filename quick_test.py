import sys
import time

if __name__ == '__main__':
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt as e:
        print('keyboard interrupt!')
        print(e)
        sys.exit(0)
