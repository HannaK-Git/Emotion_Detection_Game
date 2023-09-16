import cv2
from deepface import DeepFace
import random
import time
import datetime


def count_percentage(num_100: int, num_part: int) -> int:
    """
    Function that count what is the percent of right answers
    :param num_100: Number that represents 100%
    :param num_part: Number that represents number of right answers
    :return: Percent of right answers
    """
    return (num_part/num_100) * 100

def write_text_on_frame(frame, text: str) -> None:
    """
    Function that receives a frame and text and set the text on the frame
    :param frame: frame where should be a text
    :param text: text that should be on a frame
    :return: None
    """
    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    thickness = 2
    position = (10, 30)
    # Add the text to the frame
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness)


def save_frame_with_text(filename, frame) -> None:
    """
    Function that saves the frame with a text on it
    :param filename: name of the file
    :param frame: frame we want to save
    :return: None
    """
    # Save the frame as an image
    cv2.imwrite(filename, frame)


# Checking function
def check_input() -> int:
    """
    Function checks the input and continues to input integers till it will get the valid one
    Returns a valid number for this game
    :return: int
    """
    try_amount = int(input("Enter number of play rounds you want to play: "))
    while type(try_amount) != int or try_amount < 1 or try_amount > 20:
        try_amount = int(input("Your input is invalid. Enter number of play rounds you want to play: "))

    return try_amount


# Emotion Detection Game

def main() -> None:
    """
    It is a play for emotion detection. The computer propose to show some appropriate emotion
    that is randomly chosen from the given specter of emotions
    To show this emotion user has 5 seconds
    :return: Print the result of the game
    """

    emotions = ['anger', 'fear', 'neutral', 'sad', 'disgust', 'happy', 'surprise']

    total_score = 0
    cap = cv2.VideoCapture(0)

    # User chooses the number of attempts he wants to try
    attempts = check_input()
    # this variable will be used at the end to count the result of the game
    total_attempts = attempts

    # cycle works till the end of the attempts number
    while attempts > 0:
        # random integer that will represent one of the emotions in emotions list
        rand = random.randint(0, 6)
        # string that represents an emotion for this cycle
        target_emotion = emotions[rand]

        print(f"Show {target_emotion}")
        time.sleep(3)
        _, frame = cap.read()
        file_name = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        # Perform emotion analysis on the frame
        result = DeepFace.analyze(frame, actions=['emotion'])

        # write on frame dominant emotion and save it
        write_text_on_frame(frame, result[0]['dominant_emotion'])
        save_frame_with_text(f"{file_name}.jpg", frame)

        # Compare the detected emotion to the target emotion
        is_target_emotion = result[0]['dominant_emotion'] == target_emotion

        # Count the result and decrement attempts
        if is_target_emotion:
            total_score += 1
        attempts -= 1
        time.sleep(3)

    if total_score > total_attempts / 2:
        print("Great, you know how to express emotions!")
    else:
        print("Well, it seems to me that poker is your game!")
    print(f"Your total right answers are: {total_score} and it is {count_percentage(total_attempts, total_score)}")
    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()


# Run the code
if __name__ == "__main__":
    # try-except block that calls main function
    try:

        main()
    # if ValueError appears raise ValueError exception
    except ValueError as v:
        print(v)
    # if TypeError appears raise TypeError exception
    except TypeError as t:
        print(t)
