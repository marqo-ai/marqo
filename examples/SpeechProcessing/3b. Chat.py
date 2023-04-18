import marqo
from SpeechSearch.chatter import answer_question


def main():
    mq = marqo.Client(url="http://localhost:8882")

    index_name = "transcription-index"
    while True:
        query = input("Enter a query: ")
        answer = answer_question(
            query=query,
            limit=15,
            index=index_name,
            mq=mq,
        )
        print(answer)


if __name__ == "__main__":
    main()
