"""
Sample news articles dataset for testing the NLP pipeline
without requiring external API access.
"""

SAMPLE_ARTICLES = [
    {
        "title": "Tech Giant Unveils Revolutionary AI Assistant",
        "content": """
        Google announced the launch of its most advanced AI assistant at the annual developer 
        conference in Mountain View, California. The new Gemini Pro model demonstrates 
        unprecedented capabilities in natural language understanding and generation. 
        CEO Sundar Pichai emphasized the company's commitment to responsible AI development, 
        highlighting built-in safety features and ethical guidelines. The assistant can now 
        handle complex multi-step tasks, understand context across conversations, and provide 
        personalized responses. Industry experts predict this could revolutionize how people 
        interact with technology. Google's stock surged 5% following the announcement.
        """,
        "category": "Technology",
        "source": "TechCrunch",
        "date": "2024-10-15"
    },
    {
        "title": "Stock Markets Reach Record Highs Amid Economic Optimism",
        "content": """
        Wall Street celebrated a historic milestone as the S&P 500 closed above 5,000 points 
        for the first time. The rally was fueled by strong corporate earnings reports and 
        positive economic indicators. JPMorgan Chase and Bank of America both exceeded analyst 
        expectations, driving the financial sector higher. Federal Reserve Chair Jerome Powell's 
        recent comments about potential interest rate cuts further boosted investor confidence. 
        Trading volume reached its highest level in six months, with over $100 billion in 
        transactions. Retail investors have shown increased participation through platforms 
        like Robinhood and E*TRADE. Economists at Goldman Sachs predict continued growth 
        throughout the quarter, though some warn of potential overvaluation in tech stocks.
        """,
        "category": "Business",
        "source": "Bloomberg",
        "date": "2024-10-20"
    },
    {
        "title": "Champions League Final: Real Madrid Defeats Manchester City",
        "content": """
        Real Madrid claimed their 15th Champions League title with a thrilling 2-1 victory 
        over Manchester City at Wembley Stadium. Brazilian winger Vinicius Junior scored 
        the decisive goal in the 83rd minute, sending the 90,000 fans into a frenzy. 
        City dominated possession with 65% but couldn't convert their chances. Manager 
        Pep Guardiola expressed disappointment but praised his team's performance throughout 
        the season. Real Madrid's Carlo Ancelotti became the most successful manager in 
        Champions League history with five titles. The match attracted over 300 million 
        viewers worldwide, making it one of the most-watched sporting events of the year.
        """,
        "category": "Sports",
        "source": "ESPN",
        "date": "2024-06-01"
    },
    {
        "title": "Breakthrough in Cancer Treatment Shows Promising Results",
        "content": """
        Researchers at Johns Hopkins University have developed a revolutionary cancer treatment 
        that uses modified immune cells to target tumors. The Phase III clinical trial, involving 
        500 patients with advanced melanoma, showed a 70% success rate in tumor reduction. 
        Dr. Sarah Chen, lead researcher, described the findings as "transformative" for 
        oncology. The treatment, called CAR-T therapy, works by engineering a patient's own 
        immune cells to recognize and destroy cancer cells. Unlike traditional chemotherapy, 
        this approach has minimal side effects. The FDA is fast-tracking the approval process, 
        with potential market availability by early 2025. Pharmaceutical companies Pfizer and 
        Moderna have expressed interest in licensing the technology.
        """,
        "category": "Health",
        "source": "Nature Medicine",
        "date": "2024-09-10"
    },
    {
        "title": "Hollywood Writers Strike Ends After Historic Agreement",
        "content": """
        The Writers Guild of America announced an end to the 148-day strike after reaching 
        a landmark agreement with major studios. The deal includes significant pay increases, 
        better residuals for streaming content, and protections against AI-generated scripts. 
        Union president Meredith Stiehm called it "a victory for creative workers everywhere." 
        The strike, which began in May, disrupted production of numerous films and TV shows, 
        costing the entertainment industry an estimated $5 billion. Netflix, Disney, and Warner 
        Bros. Discovery were among the key negotiators. Late-night shows including 
        The Tonight Show and Saturday Night Live are expected to resume production immediately.
        """,
        "category": "Entertainment",
        "source": "Variety",
        "date": "2024-09-27"
    },
    {
        "title": "Climate Summit Reaches Historic Agreement on Carbon Emissions",
        "content": """
        World leaders at COP29 in Dubai reached a groundbreaking agreement to reduce global 
        carbon emissions by 50% by 2035. The accord, signed by 195 countries, includes 
        binding commitments and financial support for developing nations. UN Secretary-General 
        António Guterres praised the deal as "humanity's best chance to avoid climate 
        catastrophe." The agreement allocates $300 billion annually to help poorer countries 
        transition to renewable energy. China and the United States, the world's largest 
        emitters, committed to joint research initiatives. Environmental activists celebrated 
        outside the venue, though some criticized the timeline as insufficient. The agreement 
        also establishes new monitoring mechanisms to ensure compliance.
        """,
        "category": "Politics",
        "source": "BBC News",
        "date": "2024-11-15"
    },
    {
        "title": "Major Cybersecurity Breach Exposes Millions of User Accounts",
        "content": """
        A sophisticated cyberattack on CloudStore, a popular cloud storage provider, 
        compromised personal data of over 10 million users. The breach, discovered last 
        Tuesday, exposed names, email addresses, and encrypted passwords. Company CEO 
        Michael Torres apologized and announced enhanced security measures, including 
        mandatory two-factor authentication. Cybersecurity firm CrowdStrike is leading 
        the investigation, suggesting the attack originated from Eastern Europe. Affected 
        users are being notified and offered free credit monitoring services. The incident 
        has reignited debates about data privacy regulations, with lawmakers calling for 
        stricter penalties for companies that fail to protect user information.
        """,
        "category": "Technology",
        "source": "Reuters",
        "date": "2024-10-05"
    },
    {
        "title": "Olympic Games Paris 2024: Record-Breaking Performances",
        "content": """
        The Paris 2024 Olympics concluded with spectacular closing ceremonies at the 
        Stade de France. American gymnast Simone Biles won three gold medals, solidifying 
        her status as the greatest of all time. Jamaican sprinter Usain Bolt's protégé, 
        Marcus Thompson, set a new 100-meter world record of 9.58 seconds. China topped 
        the medal table with 40 golds, followed by the United States with 38 and Great 
        Britain with 22. The Games attracted over 3 billion viewers globally and generated 
        an estimated €10 billion in economic impact for France. Paris Mayor Anne Hidalgo 
        declared the event a "resounding success" despite initial concerns about security 
        and logistics.
        """,
        "category": "Sports",
        "source": "Olympics.com",
        "date": "2024-08-11"
    },
    {
        "title": "New Study Links Social Media Use to Teen Mental Health Crisis",
        "content": """
        A comprehensive study by Harvard Medical School found a strong correlation between 
        heavy social media use and increased rates of anxiety and depression among teenagers. 
        The research, spanning five years and involving 50,000 participants, revealed that 
        teens spending more than three hours daily on platforms like Instagram and TikTok 
        were 35% more likely to experience mental health issues. Lead researcher Dr. Emily 
        Rodriguez recommends limiting screen time and promoting real-world social interactions. 
        The findings have prompted calls for age restrictions and content moderation reforms. 
        Meta and ByteDance, parent companies of Instagram and TikTok, disputed some conclusions 
        but pledged to introduce new wellness features for younger users.
        """,
        "category": "Health",
        "source": "The Lancet",
        "date": "2024-07-22"
    },
    {
        "title": "SpaceX Successfully Launches First Commercial Moon Mission",
        "content": """
        SpaceX made history by launching the first private spacecraft to land on the Moon. 
        The Starship lunar lander, carrying scientific instruments and commercial payloads, 
        successfully touched down in the Oceanus Procellarum region. CEO Elon Musk called 
        it "a giant leap for commercial space exploration." NASA Administrator Bill Nelson 
        congratulated the team and highlighted the mission's importance for the Artemis 
        program. The spacecraft will conduct experiments on lunar soil composition and test 
        technologies for future human settlements. Japan's ispace and India's ISRO are 
        planning similar missions in the coming months, signaling a new era of international 
        and commercial lunar exploration.
        """,
        "category": "Technology",
        "source": "Space.com",
        "date": "2024-11-03"
    }
]


def get_sample_dataset():
    """
    Returns the sample dataset as a pandas DataFrame.
    Useful for batch testing and demonstrations.
    """
    import pandas as pd
    return pd.DataFrame(SAMPLE_ARTICLES)


def get_random_article():
    """Get a random article from the dataset"""
    import random
    return random.choice(SAMPLE_ARTICLES)


def get_articles_by_category(category):
    """Get all articles of a specific category"""
    return [article for article in SAMPLE_ARTICLES if article['category'] == category]


if __name__ == "__main__":
    # Demo: Print dataset info
    df = get_sample_dataset()
    print("Sample Dataset Information")
    print("=" * 50)
    print(f"Total articles: {len(df)}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())
    print(f"\nAverage content length: {df['content'].str.len().mean():.0f} characters")
    print("\nSample article:")
    print(df.iloc[0]['title'])
    print(df.iloc[0]['content'][:200] + "...")
