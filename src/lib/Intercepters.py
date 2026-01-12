# 添加请求拦截器
import os


def auth_interceptor(method, url, **kwargs):
    token = os.environ.get("GITHUB_ACCESS_TOKEN")
    headers = kwargs.setdefault("headers", {})
    headers["Authorization"] = f"Bearer {token}"
    headers["User-Agent"] = "MyApp/1.0"
    print(f"→ {method} {url}")
    return kwargs


# 添加响应拦截器
def log_response(response):
    print(f"← {response.status_code} {response.url}")
    if response.status_code == 401:
        raise Exception("Unauthorized!")
    return response
