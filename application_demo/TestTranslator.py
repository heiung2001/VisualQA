from googletrans import Translator


if __name__ == '__main__':
    # define a translate object
    translate = Translator()
    # Translate some text
    result = translate.translate('Chúng tôi là nhóm vinasupport', src='vi', dest='en')

    print(result)
    print(result.text)

    print('Test')