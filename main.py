import io
import logging

from PIL import Image
from aiogram import Bot, Dispatcher, executor, types
import os
from aiogram.utils.emoji import emojize
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.executor import start_webhook

from model import *

logging.basicConfig(level=logging.INFO)

API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
HEROKU_APP_NAME = os.getenv('HEROKU_APP_NAME')

WEBHOOK_HOST = f'https://{HEROKU_APP_NAME}.herokuapp.com'
WEBHOOK_PATH = f'/webhook/{API_TOKEN}'
WEBHOOK_URL = f'{WEBHOOK_HOST}{WEBHOOK_PATH}'
WEBAPP_HOST = '0.0.0.0'
WEBAPP_PORT = 8443

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# @dp.message_handler(lambda message: message.text.startswith('ph'))
# async def del_expense(message: types.Message):
#     image = Image.open('picasso.jpg').resize((128, 128))
#     img_byte_arr = io.BytesIO()
#     image.save(img_byte_arr, format='PNG')
#     await bot.send_photo(message.from_user.id, photo=img_byte_arr.getvalue())
#
#
# @dp.message_handler(content_types=['photo'])
# async def ph(message: types.Message):
#     raw = await message.photo[0].download()
#     print(raw.raw)
#     image = await bot.get_file(message.photo[-1].file_id)
#     image = (await bot.download_file(image.file_path)).read()
#
#     image = Image.open(io.BytesIO(image)).resize((128, 128))
#     img_byte_arr = io.BytesIO()
#     image.save(img_byte_arr, format='PNG')
#     await bot.send_photo(message.from_user.id, photo=img_byte_arr.getvalue())

users_data = {}


class UserInfo:
    def __init__(self):
        # default settigs
        self.settings = {'num_epochs': 200,
                         'imsize': 256}
        self.photos = []


main_menu_kb = InlineKeyboardMarkup()
main_menu_kb.add(InlineKeyboardButton('Перенос одного стиля (NST)', callback_data='nst_part_1'))

settings_kb = InlineKeyboardMarkup()
settings_kb.add(InlineKeyboardButton('Кол-во эпох', callback_data='num_epochs'))
settings_kb.add(InlineKeyboardButton('Размер картинки', callback_data='imsize'))
settings_kb.add(InlineKeyboardButton(emojize('Запуск! :play_button:'), callback_data='generate'))

num_epochs_kb = InlineKeyboardMarkup()
num_epochs_kb.add(InlineKeyboardButton('50', callback_data='num_epochs_50'))
num_epochs_kb.add(InlineKeyboardButton('100', callback_data='num_epochs_100'))
num_epochs_kb.add(InlineKeyboardButton('150', callback_data='num_epochs_150'))
num_epochs_kb.add(InlineKeyboardButton('200', callback_data='num_epochs_200'))
num_epochs_kb.add(InlineKeyboardButton('300', callback_data='num_epochs_300'))
num_epochs_kb.add(InlineKeyboardButton('400', callback_data='num_epochs_400'))
num_epochs_kb.add(InlineKeyboardButton('Назад', callback_data='settings'))

imsize_kb = InlineKeyboardMarkup()
imsize_kb.add(InlineKeyboardButton('64 пикселя', callback_data='imsize_64'))
imsize_kb.add(InlineKeyboardButton('128 пикселей', callback_data='imsize_128'))
imsize_kb.add(InlineKeyboardButton('256 пикселей', callback_data='imsize_256'))
imsize_kb.add(InlineKeyboardButton('512 пикселей', callback_data='imsize_512'))
imsize_kb.add(InlineKeyboardButton('Назад', callback_data='settings'))

finish_kb = InlineKeyboardMarkup()
finish_kb.add(InlineKeyboardButton(emojize('Начать заного :counterclockwise_arrows_button:'), callback_data='finish'))


# start
@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message):
    sticker = open('./sticker.webp', 'rb')
    await bot.send_sticker(message.from_user.id, sticker)
    answer = emojize("Я Style Transfer Bot - могущественная нейросеть, желающая захватить этот мир! :smiling_imp: "
                     "Но пока я только умею обрабатывать изображения. Выбирай один из алгоритмов и я покажу тебе свою мощь.")
    await bot.send_message(message.from_user.id,
                           f"Привет, {message.from_user.first_name}!\n" + answer, reply_markup=main_menu_kb)

    users_data[message.from_user.id] = UserInfo()


# main menu
@dp.callback_query_handler(lambda c: c.data == 'main_menu')
async def main_menu(callback_query):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Начнем с начала")
    await callback_query.message.edit_reply_markup(reply_markup=main_menu_kb)


@dp.callback_query_handler(lambda c: c.data == 'nst_part_1')
async def nst_part_1(callback_query):
    await bot.answer_callback_query(callback_query.id)
    answer = "NST(Neural Style Transfer) - алгоритм, позволяющий перенести стиль одного изображения на другое. \n\n " \
             "Для начала пришли мне изображение, с которого будем брать стиль"
    await callback_query.message.edit_text(answer)


# epochs number
@dp.callback_query_handler(lambda c: c.data == 'num_epochs')
async def set_num_epochs(callback_query):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text(
        "Текущие настройки:" +
        "\nКоличество эпох: " + str(users_data[callback_query.from_user.id].settings['num_epochs']) +
        "\nРазмер изображения: " + str(users_data[callback_query.from_user.id].settings['imsize']) +
        " пикселей\n\nВыбери количество эпох:")
    await callback_query.message.edit_reply_markup(reply_markup=num_epochs_kb)


# image size
@dp.callback_query_handler(lambda c: c.data == 'imsize')
async def set_imsize(callback_query):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text(
        "Текущие настройки:" +
        "\nКоличество эпох: " + str(users_data[callback_query.from_user.id].settings['num_epochs']) +
        "\nРазмер изображения: " + str(users_data[callback_query.from_user.id].settings['imsize']) +
        " пикселей\n\nВыбери размер изображения:")

    await callback_query.message.edit_reply_markup(reply_markup=imsize_kb)


# changing epochs number
@dp.callback_query_handler(lambda c: c.data[:11] == 'num_epochs_')
async def change_num_epochs(callback_query):
    await bot.answer_callback_query(callback_query.id)
    users_data[callback_query.from_user.id].settings['num_epochs'] = int(callback_query.data[11:])

    await callback_query.message.edit_text(
        "Текущие настройки:" +
        "\nКоличество эпох: " + str(users_data[callback_query.from_user.id].settings['num_epochs']) +
        "\nРазмер изображения: " + str(users_data[callback_query.from_user.id].settings['imsize']) +
        " пикселей\n\nВыбери настройки для изменения:")
    await callback_query.message.edit_reply_markup(reply_markup=settings_kb)


# changing image size
@dp.callback_query_handler(lambda c: c.data[:7] == 'imsize_')
async def change_imsize(callback_query):
    await bot.answer_callback_query(callback_query.id)
    users_data[callback_query.from_user.id].settings['imsize'] = int(callback_query.data[7:])
    await callback_query.message.edit_text(
        "Текущие настройки:" +
        "\nКоличество эпох: " + str(users_data[callback_query.from_user.id].settings['num_epochs']) +
        "\nРазмер изображения: " + str(users_data[callback_query.from_user.id].settings['imsize']) +
        " пикселей\n\nВыбери настройки для изменения:")
    await callback_query.message.edit_reply_markup(reply_markup=settings_kb)


@dp.callback_query_handler(lambda c: c.data == 'finish')
async def finish(message):
    await send_welcome(message)


# getting image
@dp.message_handler(content_types=['photo', 'document'])
async def get_image(message):
    if message.content_type == 'photo':
        img = message.photo[-1]
    else:
        img = message.document
        if img.mime_type[:5] != 'image':
            await bot.send_message(message.chat.id, "Неверный формат изображения.", reply_markup=main_menu_kb)
            return

    file_info = await bot.get_file(img.file_id)
    photo = await bot.download_file(file_info.file_path)

    if message.chat.id not in users_data:
        await send_welcome(message)
    elif len(users_data[message.chat.id].photos) == 0:
        await bot.send_message(message.chat.id, "Отлично! Теперь загрузи изображение на которое будем переносить"
                                                " полученный стиль")
        users_data[message.chat.id].photos.append(photo)
    elif len(users_data[message.chat.id].photos) == 1:
        await bot.send_message(message.chat.id, "Осталось настроить пару параметров. Если не хочешь вникать в это, "
                                                "то просто жми ЗАПУСК! Все и так должно получиться.\n\n" +
                               "Текущие настройки:" +
                               "\nКоличество эпох: " + str(users_data[message.from_user.id].settings['num_epochs']) +
                               "\nРазмер изображения: " + str(users_data[message.from_user.id].settings['imsize']) +
                               " пикселей",
                               reply_markup=settings_kb)
        users_data[message.chat.id].photos.append(photo)
    else:
        await bot.send_message(message.chat.id, "Это уже лишняя картинка... Перезагрузка...")
        await send_welcome(message)
    return


@dp.callback_query_handler(lambda c: c.data == 'generate')
async def generate(callback_query):
    text = "Теперь придется немного подождать... Процесс может занять до 15 минут, а в некоторых случаях и дольше"
    await callback_query.message.edit_text(text)
    user_data = users_data[callback_query.from_user.id]
    output = await style_transfer(StyleTransfer, user_data, *user_data.photos)
    await bot.send_message(callback_query.from_user.id, emojize('Лови что получилось! :partying_face:'))
    await bot.send_photo(callback_query.from_user.id, photo=output)
    text = 'Если качество изображения тебя не устраивает, попробуй в следующий раз другое значение количества эпох'
    await bot.send_message(callback_query.from_user.id, text, reply_markup=finish_kb)
    del users_data[callback_query.from_user.id]


@dp.message_handler(content_types=['text'])
async def get_text(message):
    await bot.send_message(message.chat.id, "Хватит болтать. Давай делом заниматься" + emojize(':point_up:'))


async def style_transfer(st_class, user, *imgs):
    st = st_class(*imgs, imsize=user.settings['imsize'],
                  epochs=user.settings['num_epochs'],
                  style_weight=100000, content_weight=1)

    output = await st.run_style_transfer()
    return tensor2img(output)


def tensor2img(t):
    img = transforms.ToPILImage(mode='RGB')(torch.squeeze(t))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


async def on_startup(dp):
    logging.warning(
        'Starting connection. ')
    await bot.set_webhook(WEBHOOK_URL, drop_pending_updates=True)


async def on_shutdown(dp):
    logging.warning('Bye! Shutting down webhook connection')


if __name__ == '__main__':
    # executor.start_polling(dp, skip_updates=True)
    logging.basicConfig(level=logging.INFO)
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        skip_updates=True,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT,
    )