from typing import Dict, Any, List, Union, TypedDict, Optional

class ContentItem(TypedDict, total=False):
    type: str
    text: Optional[str]
    image_url: Optional[str]

class Message(TypedDict):
    role: str
    content: Union[str, List[ContentItem]]

# Action constants
FINISH_WORD = "finished"
WAIT_WORD = "wait"
ERROR_WORD = "error" 