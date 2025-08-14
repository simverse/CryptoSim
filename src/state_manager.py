
import pickle
import os
import logging

class StateManager:
    """
    프로그램의 실행 상태(포지션, 미체결 주문 등)를 저장하고 불러오는 클래스
    """
    def __init__(self, config):
        # 상태를 저장할 파일 경로를 설정 파일에서 가져옵니다.
        # 설정이 없는 경우 기본값으로 'state.pkl'을 사용합니다.
        self.state_file_path = config.get('live', {}).get('state_file_path', 'state.pkl')

    def save_state(self, positions, open_orders, transactions):
        """
        현재 포지션, 미체결 주문, 거래 상세정보 상태를 파일에 저장합니다.

        Args:
            positions (dict): 현재 보유 포지션 딕셔너리
            open_orders (list): 미체결 주문 리스트
            transactions (dict): 현재 진행중인 거래 상세 딕셔너리
        """
        state = {
            'positions': positions,
            'open_orders': open_orders,
            'transactions': transactions
        }
        try:
            with open(self.state_file_path, 'wb') as f:
                pickle.dump(state, f)
            logging.info(f"현재 상태가 '{self.state_file_path}' 파일에 성공적으로 저장되었습니다.")
        except Exception as e:
            logging.error(f"상태 저장 중 오류 발생: {e}")

    def load_state(self):
        """
        파일에서 상태를 불러옵니다.

        Returns:
            tuple: (positions, open_orders, transactions) 튜플
        """
        if not os.path.exists(self.state_file_path):
            return {}, [], {}

        try:
            with open(self.state_file_path, 'rb') as f:
                state = pickle.load(f)
            logging.info(f"'{self.state_file_path}' 파일에서 상태를 성공적으로 불러왔습니다.")
            return state.get('positions', {}), state.get('open_orders', []), state.get('transactions', {})
        except Exception as e:
            logging.error(f"상태 불러오기 중 오류 발생: {e}. 새로운 상태로 시작합니다.")
            return {}, [], {}
