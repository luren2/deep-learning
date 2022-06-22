import requests    # 发送请求
from bs4 import BeautifulSoup # 解析网页源代码

# 发送请求到服务器
resp = requests.get("https://www.53pic.com/bizhi/yingshi/")
resp.encoding = 'utf-8'
# print(resp.text)
# 解析html
main_page = BeautifulSoup(resp.text, "html.parser")

# print(main_page)
# 从页面中找到某些东西
# find() 找一个
# find_all() 找所有
result = main_page.find("div", attrs={"class": "all-work-list"}).find_all("a", attrs={"class": "card-img-hover"})
# print(result)
n=1
for a in result:
    # print('https://www.53pic.com/' + a.get("href"))

    # 发送请求到子页面
    href = 'https://www.53pic.com/' + a.get("href")
    # print(href)
    resp1 = requests.get(href)
    resp1.encoding = 'utf-8'
    child_page = BeautifulSoup(resp1.text, "html.parser")
    # 找到图片的真实路径
    src = child_page.find("div", attrs={"class": "cl mbv"}).find("img").get("src")
    # print(src)
    # 发送请求到服务器，把图片保存到本地
    # 创建文件
    f = open("C://Users\pc\Desktop\爬取\图片\图_%s.jpg" % n, mode="wb")  # wb表示写入的内容是非文本文件
    f.write(requests.get(src).content)   #向外拿出图片的数据 不是文本信息 必须用content来拿
    print("已下载%s张图片" % n)
    n += 1

  # print(result)

