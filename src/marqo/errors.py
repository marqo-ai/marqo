import json

from requests import Response

class MarqoError(Exception):
    """Generic class for Marqo error handling"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f'MarqoError. Error message: {self.message}'


class MarqoApiError(MarqoError):
    """Error sent by Marqo API"""

    def __init__(self, error: str, request: Response) -> None:
        self.status_code = request.status_code
        self.code = None
        self.link = None
        self.type = None

        if request.text:
            json_data = json.loads(request.text)
            self.message = json_data
            self.code = json_data.get('status')
            self.link = ''
            self.type = ''
            if 'error' in json_data and 'root_cause' in json_data["error"]\
                    and len(json_data.get('error').get('root_cause')) > 0:
                self.type = json_data.get('error').get('root_cause')[0].get('type')
        else:
            self.message = error
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.code and self.link:
            return f'MarqoApiError. Error code: {self.code}. Error message: {self.message} Error documentation: {self.link} Error type: {self.type}'

        return f'MarqoApiError. {self.message}'


class MarqoNonNeuralIndexError(MarqoError):
    """Error trying to use a non-neural index like a neural one"""

    def __str__(self) -> str:
        return f'MarqoCommunicationError, {self.message}'


class MarqoCommunicationError(MarqoError):
    """Error when connecting to Marqo"""

    def __str__(self) -> str:
        return f'MarqoCommunicationError, {self.message}'


class MarqoTimeoutError(MarqoError):
    """Error when Marqo operation takes longer than expected"""

    def __str__(self) -> str:
        return f'MarqoTimeoutError, {self.message}'