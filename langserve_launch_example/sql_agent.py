from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
import os

db = SQLDatabase.from_uri(os.getenv("TONO_POSTGRES_URL"))

llm = ChatOpenAI(
    model="defog/sqlcoder-7b-2",
    base_url=os.getenv("SQL_LLM_URL"),
    api_key=os.getenv("RUNPOD_API_KEY"),
    temperature=0,
)

prompt = PromptTemplate.from_template(
    """ 
### Instructions:
Your task is to convert a question into a SQL query, given a Postgres database schema.
Adhere to these rules:
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
- When creating a ratio, always cast the numerator as float

### Input:
Generate a SQL query that answers the question `{query}`.

This query will run on a database whose schema is represented in this string:

CREATE TABLE "public"."discord_gateway_events" (
    "id" int4 NOT NULL DEFAULT nextval('discord_gateway_events_id_seq'::regclass), -- Unique ID for each event
    "type" text, -- type of this event. messageCreate type is when new message is created. guildMemberAdd type is when new member is added
    "content" text, -- content message
    "author_id" varchar, -- id of the event's author
    "author_username" varchar, -- username of the event's author
    "event_time" timestamptz, -- the time event created
     PRIMARY KEY ("id")
);

- messageCreate: type of event when a new message is created. Any questions related to messages or chat should be based on this type
- guildMemberAdd: type of event when a new member is added

- always return author_username when questions related to user, people

### Response:
Based on your instructions, here is the SQL query I have generated to answer the question `{query}`:
```sql
"""
)


execute_query = QuerySQLDataBaseTool(db=db)
write_query = prompt | llm | StrOutputParser()

sql_agent = write_query | execute_query

# result = sql_agent.invoke(
#     {"query": "What is the total number of messages sent by each user yesterday?"})

# print(result)
