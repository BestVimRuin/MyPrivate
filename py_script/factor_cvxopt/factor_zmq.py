import zmq
import json
import logging
import os
from factor_cvxopt import factor
from factor_cvxopt import factor_copy
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

log_path = r'D:\gitee\htsc\factor_cvxopt\factor_cvxopt.log'
logging.basicConfig(filename='%s' % log_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


logger.info('zmq is running')
while True:
    message = socket.recv()
    # Do some 'work'
    logger.info('starting...')
    try:
        logger.info('func run is starting...')
        result = factor_copy.run(message)
        logger.info('func run has done...')
        result_json = json.dumps(result)
        socket.send_string(result_json)
        logger.info('zmq has sent successful---------------------------------')
    except Exception as err:
        socket.send(b'fail')
        logger.error('zmq error...')
    