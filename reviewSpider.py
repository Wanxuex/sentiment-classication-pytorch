from scrapy.spiders import Spider

class reviewSpider(Spider):
    name = ""
    start_urls = ['https://movie.douban.com/subject/34463197/comments?status=P']

    def parse(self, response):
        titles = response.xpath('//a[@class="post-title-link"]/text()').extract()
        for title in titles:
            print(title.strip())
