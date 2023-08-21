import scrapy
from datetime import datetime
import json


class QuotesSpider(scrapy.Spider):
    name = "quotes"
    start_urls = [
        "https://quotes.toscrape.com/tag/humor/",
    ]

    def parse(self, response):
        for quote in response.css("div.quote"):
            yield {
                "author": quote.xpath("span/small/text()").get(),
                "text": quote.css("span.text::text").get(),
            }

        next_page = response.css('li.next a::attr("href")').get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)


class GoogSpider(scrapy.Spider):
    name = "google"
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
        'LOG_LEVEL': 'INFO',
        'CONCURRENT_REQUESTS_PER_DOMAIN': 10,
        'RETRY_TIMES': 5,
        'DOWNLOAD_DELAY': 2,
        'COOKIES_ENABLED': False}
    start_urls = [
        "https://www.google.com/search?q=marija+kojic+liquid+biopsy"
        ]

    def parse(self, response):
        di = json.loads(response.text)
        pos = response.meta['pos']
        dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for result in di['organic_results']:
            title = result['title']
            snippet = result['snippet']
            link = result['link']
            item = {'title': title, 'snippet': snippet, 'link': link, 'position': pos, 'date': dt}
            pos += 1
            yield item
        next_page = di['pagination']['nextPageUrl']
        if next_page:
            yield scrapy.Request(next_page, callback=self.parse, meta={'pos': pos})