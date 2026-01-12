import requests
from typing import Callable, Optional

from lib.Intercepters import auth_interceptor, log_response


class InterceptedSession:
    def __init__(self):
        self.session = requests.Session()
        self.request_interceptors = []
        self.response_interceptors = []

    def add_request_interceptor(self, func: Callable):
        """添加请求拦截器，func 接收 (method, url, **kwargs) 并返回修改后的 kwargs"""
        self.request_interceptors.append(func)

    def add_response_interceptor(self, func: Callable):
        """添加响应拦截器，func 接收 response 并可返回修改后的 response"""
        self.response_interceptors.append(func)

    def _apply_request_interceptors(self, method, url, **kwargs):
        for interceptor in self.request_interceptors:
            kwargs = interceptor(method=method, url=url, **kwargs) or kwargs
        return kwargs

    def _apply_response_interceptors(self, response):
        for interceptor in self.response_interceptors:
            result = interceptor(response)
            if result is not None:
                response = result
        return response

    def request(self, method, url, **kwargs):
        kwargs = self._apply_request_interceptors(method, url, **kwargs)
        response = self.session.request(method, url, **kwargs)
        response = self._apply_response_interceptors(response)
        return response

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def put(self, url, **kwargs):
        return self.request("PUT", url, **kwargs)

    def delete(self, url, **kwargs):
        return self.request("DELETE", url, **kwargs)


def get_intercepted_session() -> InterceptedSession:
    client = InterceptedSession()
    # request interceptors
    client.add_request_interceptor(auth_interceptor)

    # response interceptors
    client.add_response_interceptor(log_response)
    return client
