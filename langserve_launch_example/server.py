from fastapi import FastAPI
from langserve import add_routes
from langserve_launch_example.runpod_wrapper import sql_agent, write_query

from langserve_launch_example.chain import get_chain

app = FastAPI(title="LangServe Launch Example")

add_routes(app, sql_agent)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
