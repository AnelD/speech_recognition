from pydantic import BaseModel


class Content(BaseModel):
    fileName: str
    text: str


class Message(BaseModel):
    type: str
    message: Content
