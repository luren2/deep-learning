import requests
from bs4 import BeautifulSoup # 解析网页源代码

# url = 'https://music.163.com/discover/playlist'
url = 'https://y.qq.com/n/ryqq/category'
headers = {
      'user-agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0'
}

resp = requests.get(url=url, headers=headers)
# resp.raise_for_status()
# print(resp.text)

main_page = BeautifulSoup(resp.text, "html.parser")
# print(main_page)
result = main_page.find("div", attrs={"class": "playlist__item_box"}).find_all("a")
print(result)

n=1
for a in result:
    # print('https://www.53pic.com/' + a.get("href"))

    #发送请求到子页面
    href = 'https://y.qq.com' + a.get("href")
    # href = a.get("href")
    print(href)
    # resp1 = requests.get(url=href, headers=headers)
    resp1=requests.get(href)
    resp1.encoding = 'utf-8'
    child_page = BeautifulSoup(resp1.text, "html.parser")
    # 找到图片的真实路径
    src = child_page.find("span", attrs={"class": "data__cover"}).find("img").get("src")
    print(src)
    # 发送请求到服务器，把图片保存到本地
    # 创建文件
    # f = open("C://Users\pc\Desktop\爬取\图片\图_%s.jpg" % n, mode="wb")  # wb表示写入的内容是非文本文件
    # f.write(requests.get(src).content)   #向外拿出图片的数据 不是文本信息 必须用content来拿
    # print("已下载%s张图片" % n)
    # n += 1


