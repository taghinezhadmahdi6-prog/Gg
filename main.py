import logging
import asyncio
import io
import base64
import json
import re
from datetime import datetime

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

import google.generativeai as genai
from PIL import Image
from openai import AsyncOpenAI

# ---------------- تنظیمات ----------------
# 1. توکن ربات تلگرام

TELEGRAM_TOKEN = "8229826436:AAGBM8IxFw6zHqhB38b3OmjqrsDprCfKpPA"

# 2. تنظیمات گوگل جمینی (سرویس اصلی)
GOOGLE_API_KEY = "AIzaSyAuvryviPqsfFi8jdUF7fo9nU-eAAqpP_A"
GEMINI_MODEL_ID = "gemini-flash-latest"  # پیشنهاد: gemini-1.5-flash یا gemini-1.5-pro

# 3. تنظیمات Clarifai (سرویس جایگزین)
CLARIFAI_API_KEY = "c21e5e3be76e452ea4c2ffea19b32d58"
CLARIFAI_BASE_URL = "https://api.clarifai.com/v2/ext/openai/v1"
CLARIFAI_MODEL_ID = "https://clarifai.com/openai/chat-completion/models/o4-mini/versions/efcf58b9be9243ffb6e4032e97a40040"
# ----------------------------------------

# ✅ کانفیگ Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# کلاینت Async برای Clarifai
clarifai_client = AsyncOpenAI(
    api_key=CLARIFAI_API_KEY,
    base_url=CLARIFAI_BASE_URL,
)

# حافظه موقت
user_invoices = {}
user_reports = {}
MAX_REPORTS_PER_USER = 5

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ---------- ابزارهای کمکی ----------

PERSIAN_DIGITS_MAP = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

def to_persian_digits(s) -> str:
    if s is None: return "۰"
    return "{:,}".format(int(float(s))).translate(PERSIAN_DIGITS_MAP)

def encode_image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def try_extract_json(text: str):
    if not text: return None
    text = text.strip()
    # تلاش برای یافتن JSON در بین توضیحات احتمالی مدل
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    try:
        return json.loads(text)
    except:
        return None

# ---------- تولید گزارش HTML ----------

def build_html_report(report_obj: dict, raw_fallback_text: str) -> bytes:
    """
    تولید فایل HTML زیبا با پشتیبانی از راست‌چین (RTL) و استایل‌های CSS
    """
    if not isinstance(report_obj, dict):
        html_content = f"""
        <html><body>
        <h1>خطا در پردازش</h1>
        <p>خروجی خام مدل:</p>
        <pre>{raw_fallback_text}</pre>
        </body></html>
        """
        return html_content.encode("utf-8")

    invoices = report_obj.get("invoices", [])
    grand_total_payable = 0
    
    # محاسبه جمع کل نهایی از روی مبالغ قابل پرداخت
    for inv in invoices:
        fin = inv.get("financials", {})
        grand_total_payable += int(fin.get("payable_amount", 0))

    # شروع ساخت HTML
    html_parts = []
    html_parts.append("""
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <title>گزارش فاکتورها</title>
        <style>
            body { font-family: 'Tahoma', 'Segoe UI', sans-serif; background-color: #f4f4f9; padding: 20px; color: #333; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { text-align: center; color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
            .invoice-box { border: 1px solid #ddd; border-radius: 8px; margin-bottom: 20px; overflow: hidden; }
            .invoice-header { background-color: #34495e; color: white; padding: 10px 15px; display: flex; justify-content: space-between; align-items: center; }
            .invoice-header h3 { margin: 0; font-size: 1.1em; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 0; }
            th, td { padding: 10px; text-align: center; border-bottom: 1px solid #eee; font-size: 0.9em; }
            th { background-color: #f8f9fa; color: #555; font-weight: bold; }
            tr:last-child td { border-bottom: none; }
            .financial-summary { background-color: #ecf0f1; padding: 15px; border-top: 1px solid #ddd; display: flex; flex-wrap: wrap; gap: 15px; justify-content: flex-end; }
            .fin-item { background: white; padding: 5px 10px; border-radius: 5px; border: 1px solid #ccc; font-size: 0.9em; }
            .payable { background-color: #27ae60; color: white; font-weight: bold; border: none; font-size: 1.1em; }
            .grand-total-box { background-color: #2c3e50; color: white; text-align: center; padding: 20px; border-radius: 10px; margin-top: 30px; font-size: 1.5em; }
            .badge { background: #e74c3c; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📄 گزارش صورت‌حساب‌ها</h1>
    """)

    for idx, inv in enumerate(invoices, 1):
        inv_no = inv.get("invoice_no", "---")
        items = inv.get("items", [])
        fin = inv.get("financials", {})
        
        # هدر فاکتور
        html_parts.append(f"""
            <div class="invoice-box">
                <div class="invoice-header">
                    <h3>فاکتور شماره {to_persian_digits(idx)}</h3>
                    <span style="font-size:0.9em; opacity:0.9;">کد سفارش: {inv_no}</span>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>ردیف</th>
                            <th>نام کالا</th>
                            <th>تعداد</th>
                            <th>فی (تومان)</th>
                            <th>قیمت کل</th>
                        </tr>
                    </thead>
                    <tbody>
        """)
        
        # ردیف کالاها
        for i, item in enumerate(items, 1):
            html_parts.append(f"""
                        <tr>
                            <td>{to_persian_digits(i)}</td>
                            <td style="text-align:right;">{item.get('name', '')}</td>
                            <td>{to_persian_digits(item.get('qty', 0))}</td>
                            <td>{to_persian_digits(item.get('unit_price', 0))}</td>
                            <td>{to_persian_digits(item.get('total_price', 0))}</td>
                        </tr>
            """)
            
        # بخش مالی پایین فاکتور
        sum_items = to_persian_digits(fin.get('sum_items', 0))
        shipping = to_persian_digits(fin.get('shipping', 0))
        discount = to_persian_digits(fin.get('discount', 0))
        payable = to_persian_digits(fin.get('payable_amount', 0))

        html_parts.append(f"""
                    </tbody>
                </table>
                <div class="financial-summary">
                    <div class="fin-item">جمع اقلام: {sum_items}</div>
                    <div class="fin-item">هزینه ارسال: {shipping}</div>
                    <div class="fin-item" style="color:#e74c3c">تخفیف: {discount}</div>
                    <div class="fin-item payable">قابل پرداخت: {payable} تومان</div>
                </div>
            </div>
        """)

    # جمع کل نهایی
    html_parts.append(f"""
            <div class="grand-total-box">
                مبلغ کل نهایی: {to_persian_digits(grand_total_payable)} تومان
            </div>
            <div style="text-align:center; margin-top:20px; color:#999; font-size:0.8em;">
                زمان گزارش: {datetime.now().strftime('%Y/%m/%d - %H:%M')}
            </div>
        </div>
    </body>
    </html>
    """)

    return "".join(html_parts).encode("utf-8")

def build_txt_report(report_obj: dict, raw_text: str) -> bytes:
    """گزارش متنی ساده برای پیش‌نمایش سریع"""
    if not isinstance(report_obj, dict):
        return raw_text.encode('utf-8')
    
    lines = ["=== گزارش خلاصه ==="]
    grand_sum = 0
    for inv in report_obj.get("invoices", []):
        pay = inv.get("financials", {}).get("payable_amount", 0)
        grand_sum += int(pay)
        lines.append(f"سفارش: {inv.get('invoice_no')} | مبلغ: {to_persian_digits(pay)} تومان")
    
    lines.append("-" * 20)
    lines.append(f"جمع کل: {to_persian_digits(grand_sum)} تومان")
    return "\n".join(lines).encode('utf-8')

async def send_report_files(update: Update, txt_bytes: bytes, html_bytes: bytes, txt_name: str, html_name: str):
    await update.message.reply_document(
        document=io.BytesIO(txt_bytes),
        filename=txt_name,
        caption="📄 خلاصه متنی"
    )
    await update.message.reply_document(
        document=io.BytesIO(html_bytes),
        filename=html_name,
        caption="🌐 گزارش کامل و گرافیکی (HTML)"
    )

def store_user_report(user_id: int, txt_bytes: bytes, html_bytes: bytes, txt_name: str, html_name: str):
    if user_id not in user_reports:
        user_reports[user_id] = []

    user_reports[user_id].insert(0, {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "txt_bytes": txt_bytes,
        "html_bytes": html_bytes,
        "txt_name": txt_name,
        "html_name": html_name,
    })
    user_reports[user_id] = user_reports[user_id][:MAX_REPORTS_PER_USER]

# ---------- پردازش با سرویس‌ها ----------

async def process_with_gemini(images, prompt):
    contents = [prompt]
    contents.extend(images)
    def _call():
        model = genai.GenerativeModel(GEMINI_MODEL_ID)
        resp = model.generate_content(contents)
        return resp.text
    return await asyncio.to_thread(_call)

async def process_with_clarifai(images, prompt):
    messages_content = [{"type": "text", "text": prompt}]
    for img in images:
        base64_image = encode_image_to_base64(img)
        messages_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            }
        })
    response = await clarifai_client.chat.completions.create(
        model=CLARIFAI_MODEL_ID,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": messages_content}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

# ---------- هندلرها ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_invoices[user_id] = []

    keyboard = [
        [KeyboardButton("✅ محاسبه و گزارش نهایی")],
        [KeyboardButton("📁 گزارش‌های قبلی")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    await update.message.reply_text(
        "سلام! سیستم حسابداری هوشمند آماده است 🚀\n"
        "📸 عکس‌های فاکتور را بفرستید.\n"
        "🔚 در آخر دکمه «محاسبه و گزارش نهایی» را بزنید.",
        reply_markup=reply_markup
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in user_invoices:
        user_invoices[user_id] = []

    msg = await update.message.reply_text("📥 در حال دریافت تصویر...")

    try:
        photo_file = await update.message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()
        img = Image.open(io.BytesIO(image_bytes))

        user_invoices[user_id].append(img)
        count = len(user_invoices[user_id])

        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=msg.message_id,
            text=f"✅ فاکتور {count} ذخیره شد."
        )
    except Exception as e:
        await update.message.reply_text(f"❌ خطا: {e}")

async def send_previous_reports(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    reports = user_reports.get(user_id, [])
    if not reports:
        await update.message.reply_text("❌ گزارشی موجود نیست.")
        return

    await update.message.reply_text(f"📁 ارسال {len(reports)} گزارش آخر...")
    for idx, r in enumerate(reports, start=1):
        caption = f"گزارش #{idx} | {r['created_at']}"
        await update.message.reply_document(
            document=io.BytesIO(r["html_bytes"]),
            filename=r["html_name"],
            caption=f"🌐 {caption}"
        )

async def process_all_invoices(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if user_id not in user_invoices or not user_invoices[user_id]:
        await update.message.reply_text("❌ عکسی برای پردازش وجود ندارد.")
        return

    images = user_invoices[user_id]
    await update.message.reply_text(f"⏳ در حال پردازش {len(images)} فاکتور با تمرکز بر «مبلغ قابل پرداخت»...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # ✅ پرامپت دقیق برای استخراج مبلغ نهایی
    prompt_text = (
        "You are an expert accountant processing Iranian invoices.\n"
        "Return ONLY valid JSON.\n\n"
        "Analyze these invoice images and extract data strictly matching this structure:\n"
        "{\n"
        '  "invoices": [\n'
        "    {\n"
        '      "invoice_no": "شماره سفارش",\n'
        '      "items": [\n'
        '        {"name": "Item Name", "qty": 1, "unit_price": 1000, "total_price": 1000}\n'
        "      ],\n"
        '      "financials": {\n'
        '         "sum_items": 1000,   // جمع قیمت کالاها\n'
        '         "shipping": 0,       // هزینه ارسال\n'
        '         "discount": 0,       // سود شما/تخفیف\n'
        '         "payable_amount": 1000 // مبلغ قابل پرداخت (مهم‌ترین عدد)\n'
        "      }\n"
        "    }\n"
        "  ],\n"
        '  "notes": "Any warnings"\n'
        "}\n\n"
        "RULES:\n"
        "1. Identify 'مبلغ قابل پرداخت' (Payable Amount) carefully. It is usually at the bottom left or highlighted.\n"
        "2. Convert all Persian numbers to English integers.\n"
        "3. If multiple invoices are in one image or across multiple images, separate them in the 'invoices' list."
    )

    result_text = ""
    source_used = ""

    # 1) Gemini
    try:
        logging.info(f"User {user_id}: Trying Gemini...")
        result_text = await process_with_gemini(images, prompt_text)
        source_used = "Google Gemini"
    except Exception as e:
        logging.error(f"Gemini Error: {e}")
        await update.message.reply_text(f"⚠️ جمینی پاسخ نداد. تلاش با سرور دوم...")
        
        # 2) Clarifai
        try:
            logging.info(f"User {user_id}: Trying Clarifai...")
            result_text = await process_with_clarifai(images, prompt_text)
            source_used = "Clarifai AI"
        except Exception as e2:
            logging.error(f"Clarifai Error: {e2}")
            await update.message.reply_text(f"❌ خطا در پردازش: {e2}")
            return

    report_obj = try_extract_json(result_text)

    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_name = f"invoice_{user_id}_{now_tag}.txt"
    html_name = f"invoice_{user_id}_{now_tag}.html"

    # تولید فایل‌ها (HTML جایگزین اکسل شد)
    txt_bytes = build_txt_report(report_obj, result_text)
    html_bytes = build_html_report(report_obj, result_text)

    store_user_report(user_id, txt_bytes, html_bytes, txt_name, html_name)
    
    # محاسبه جمع کل برای نمایش در چت
    total_payable = 0
    if isinstance(report_obj, dict):
        for inv in report_obj.get("invoices", []):
            total_payable += int(inv.get("financials", {}).get("payable_amount", 0))

    await update.message.reply_text(
        f"📊 گزارش توسط {source_used} آماده شد.\n"
        f"💰 **جمع کل قابل پرداخت:** {to_persian_digits(total_payable)} تومان\n"
        f"📎 فایل HTML (گرافیکی) ارسال شد:"
    )

    await send_report_files(update, txt_bytes, html_bytes, txt_name, html_name)
    user_invoices[user_id] = []

if __name__ == '__main__':
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.Regex(r'^✅'), process_all_invoices))
    application.add_handler(MessageHandler(filters.Regex(r'^📁'), send_previous_reports))

    print("ربات با خروجی HTML روشن شد...")
    application.run_polling()
