# -*- coding:utf-8 -*-
# @Time     :2019/5/21 14:12
# @author   :ding

"""
线程生成器
"""
from threading import Thread
from queue import Queue


class ThreadedGenerator(object):
    def __init__(self, iterator, sentinel=object(), queue_maxsize=0, daemon=False):
        self._iterator = iterator
        self._sentinel = sentinel
        self._queue = Queue(maxsize=queue_maxsize)
        self._thread = Thread(
            name=repr(iterator),
            target=self._run
        )
        self._thread.daemon = daemon
        self._started = False

    def __repr__(self):
        return "ThreadedGenerator({!r})".format(self._iterator)

    def _run(self):
        try:
            for value in self._iterator:
                if not self._started:
                    return
                self._queue.put(value)
        finally:
            self._queue.put(self._sentinel)

    def close(self):
        self._started = False
        try:
            while True:
                self._queue.get(timeout=5)
        except KeyboardInterrupt as e:
            raise e
        except Exception:
            pass

    def __iter__(self):
        self._started = True
        self._thread.start()
        for value in iter(self._queue.get, self._sentinel):
            yield value
        self._thread.join()
        self._started = False

    def __next__(self):
        if not self._started:
            self._started = True
            self._thread.start()
        value = self._queue.get(timeout=5)
        if value == self._sentinel:
            raise StopIteration()
        return value


def test():
    def gene():
        i = 0
        while True:
            yield i
            i += 1

    t = gene()
    test = ThreadedGenerator(t)

    for _ in range(30):
        print(next(test))

    test.close()


if __name__ == '__main__':
    test()
