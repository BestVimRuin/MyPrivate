import tensorflow as tf
import threading
import numpy as np
import time



def my_loop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.01:
            print(worker_id, '被停止了')
            coord.request_stop()
        else:
            print('当前线程ID：',worker_id)

        time.sleep(10)

coord = tf.train.Coordinator()
threads = [
    threading.Thread(target=my_loop,args=(coord,i)) for i in range(20)
]
for t in threads:
    t.start()
coord.join(threads)



