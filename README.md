# MySQL Query Automation using Google PaLM and LangChain

This project demonstrates how to connect to a MySQL database and run SQL queries using the Google PaLM large language model (LLM) via LangChain. The project includes setting up the LLM, connecting to the database, and performing various SQL operations with enhanced query generation and error handling through few-shot learning and semantic similarity-based example selection.

## Prerequisites

- Python 3.7+
- MySQL database
- Google Cloud API Key for Google PaLM
- Required Python packages:
  - langchain
  - pymysql
  - sqlalchemy
  - sentence-transformers
  - chromadb

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. **Install the required Python packages:**

    ```bash
    pip install langchain pymysql sqlalchemy sentence-transformers chromadb
    ```

3. **Set up your MySQL database:**

    Make sure your MySQL server is running and accessible. Create the `atliq_tshirts` database and the required tables:

    ```sql
    CREATE TABLE discounts (
        discount_id INTEGER NOT NULL AUTO_INCREMENT, 
        t_shirt_id INTEGER NOT NULL, 
        pct_discount DECIMAL(5, 2), 
        PRIMARY KEY (discount_id), 
        CONSTRAINT discounts_ibfk_1 FOREIGN KEY(t_shirt_id) REFERENCES t_shirts (t_shirt_id), 
        CONSTRAINT discounts_chk_1 CHECK ((`pct_discount` between 0 and 100))
    ) COLLATE utf8mb4_0900_ai_ci DEFAULT CHARSET=utf8mb4 ENGINE=InnoDB;

    CREATE TABLE t_shirts (
        t_shirt_id INTEGER NOT NULL AUTO_INCREMENT, 
        brand ENUM('Van Huesen','Levi','Nike','Adidas') NOT NULL, 
        color ENUM('Red','Blue','Black','White') NOT NULL, 
        size ENUM('XS','S','M','L','XL') NOT NULL, 
        price INTEGER, 
        stock_quantity INTEGER NOT NULL, 
        PRIMARY KEY (t_shirt_id), 
        CONSTRAINT t_shirts_chk_1 CHECK ((`price` between 10 and 50))
    ) COLLATE utf8mb4_0900_ai_ci DEFAULT CHARSET=utf8mb4 ENGINE=InnoDB;
    ```

## Configuration

1. **Set up your Google PaLM API key:**

    Replace `'your_api_key'` with your actual Google PaLM API key in the code:

    ```python
    api_key = 'your_api_key'
    ```

2. **Database connection settings:**

    Update the database connection settings in the code:

    ```python
    db_user = "root"
    db_password = "root"
    db_host = "localhost"
    db_name = "atliq_tshirts"
    ```

## Usage

1. **Initialize the LLM and database connection:**

    ```python
    from langchain.llms import GooglePalm
    from langchain.utilities import SQLDatabase
    from langchain_experimental.sql import SQLDatabaseChain

    api_key = 'your_api_key'
    llm = GooglePalm(google_api_key=api_key, temperature=0.2)

    db_user = "root"
    db_password = "root"
    db_host = "localhost"
    db_name = "atliq_tshirts"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}", sample_rows_in_table_info=3)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    ```

2. **Perform queries:**

    ```python
    # Example queries
    qns1 = db_chain.run("How many t-shirts do we have left for Nike in extra small size and white color?")
    qns2 = db_chain.run("SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'")
    qns3 = db_chain.run("SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from (select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi' group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id")
    ```

3. **Implement few-shot learning for better query generation:**

    ```python
    # Few-shot learning examples
    few_shots = [
        {'Question': "How many t-shirts do we have left for Nike in XS size and white color?",
         'SQLQuery': "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'XS'",
         'SQLResult': "Result of the SQL query",
         'Answer': qns1},
        {'Question': "How much is the total price of the inventory for all S-size t-shirts?",
         'SQLQuery': "SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'",
         'SQLResult': "Result of the SQL query",
         'Answer': qns2},
        {'Question': "If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue our store will generate (post discounts)?",
         'SQLQuery': """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
                        (select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
                        group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id""",
         'SQLResult': "Result of the SQL query",
         'Answer': qns3}
    ]
    ```

    ```python
    # Create semantic similarity based example selector
    from langchain.prompts import SemanticSimilarityExampleSelector
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    example_selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore, k=2)
    ```

4. **Set up the prompt template and execute queries:**

    ```python
    from langchain.prompts import FewShotPromptTemplate
    from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
    from langchain.prompts.prompt import PromptTemplate

    example_prompt = PromptTemplate(input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
                                    template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}")

    few_shot_prompt = FewShotPromptTemplate(example_selector=example_selector, example_prompt=example_prompt,
                                            prefix=mysql_prompt, suffix=PROMPT_SUFFIX, input_variables=["input", "table_info", "top_k"])

    new_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)

    # Execute queries with few-shot learning
    new_chain.run("How many white color Levi's shirt I have?")
    new_chain.run("How much is the price of the inventory for all small size t-shirts?")
    new_chain.run("How much is the price of all white color Levi t-shirts?")
    new_chain.run("If we have to sell all the Nike’s T-shirts today with discounts applied. How much revenue our store will generate (post discounts)?")
    new_chain.run('How much revenue our store will generate by selling all Van Heusen T-shirts without discount?')
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
