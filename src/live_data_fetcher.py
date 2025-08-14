import asyncio
import json
import logging
import threading
import websockets
from queue import Queue

class LiveDataFetcher:
    """
    OKX 웹소켓을 통해 실시간 시세 데이터를 수신하는 클래스
    """
    def __init__(self, config):
        self.config = config
        self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"
        self.symbol = config['trading']['symbol'].replace('/', '-').split(':')[0]
        self.channel = f"tickers"
        self.data_queue = Queue() # 수신된 데이터를 저장할 큐
        self.stop_event = threading.Event()
        self.ws_thread = threading.Thread(target=self._run_websocket)

    async def _subscribe(self, ws):
        """웹소켓 채널을 구독합니다."""
        args = [{
            "channel": self.channel,
            "instId": self.symbol
        }]
        sub_message = json.dumps({"op": "subscribe", "args": args})
        await ws.send(sub_message)
        logging.info(f"'{self.channel}' 채널 구독: {self.symbol}")

    async def _handle_messages(self, ws):
        """웹소켓으로부터 메시지를 수신하고 처리합니다."""
        while not self.stop_event.is_set():
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(message)
                if 'data' in data:
                    self.data_queue.put(data['data'][0]) # 큐에 데이터 추가
                elif 'event' in data and data['event'] == 'error':
                    logging.error(f"웹소켓 오류 수신: {data}")
                # PING/PONG 처리 (OKX는 30초마다 PING 전송)
                elif 'op' in data and data['op'] == 'pong':
                    pass # 특별한 처리 불필요
            except asyncio.TimeoutError:
                # 10초간 메시지 없으면 PING 전송으로 연결 유지 확인
                try:
                    await ws.send('ping')
                except websockets.exceptions.ConnectionClosed:
                    logging.warning("웹소켓 연결이 끊어졌습니다. 재연결을 시도합니다.")
                    break # 내부 루프를 빠져나가 재연결 로직으로 이동
            except websockets.exceptions.ConnectionClosed as e:
                logging.warning(f"웹소켓 연결이 끊어졌습니다 ({e}). 재연결을 시도합니다.")
                break
            except Exception as e:
                logging.error(f"메시지 처리 중 예상치 못한 오류 발생: {e}")
                break

    async def _websocket_client(self):
        """웹소켓 클라이언트를 실행하고 자동 재연결을 지원합니다."""
        while not self.stop_event.is_set():
            try:
                async with websockets.connect(self.ws_url) as ws:
                    await self._subscribe(ws)
                    await self._handle_messages(ws)
            except Exception as e:
                logging.error(f"웹소켓 연결 실패: {e}. 5초 후 재시도합니다.")
                await asyncio.sleep(5)

    def _run_websocket(self):
        """새로운 이벤트 루프에서 웹소켓 클라이언트를 실행합니다."""
        asyncio.run(self._websocket_client())

    def start(self):
        """웹소켓 수신을 별도의 스레드에서 시작합니다."""
        self.stop_event.clear()
        self.ws_thread.start()
        logging.info("실시간 데이터 수신 스레드를 시작합니다.")

    def stop(self):
        """웹소켓 수신을 중지합니다."""
        self.stop_event.set()
        if self.ws_thread.is_alive():
            self.ws_thread.join() # 스레드가 완전히 종료될 때까지 대기
        logging.info("실시간 데이터 수신 스레드를 중지했습니다.")

    def get_latest_data(self):
        """큐에서 가장 최근의 데이터를 가져옵니다."""
        if not self.data_queue.empty():
            return self.data_queue.get()
        return None