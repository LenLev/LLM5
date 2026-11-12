"""Build a short Russian report from actual experiment files."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

root=Path(__file__).parent
s=json.loads((root/'report/summary.json').read_text(encoding='utf-8'))
eda=json.loads((root/'data/eda.json').read_text(encoding='utf-8'))
font='C:/Windows/Fonts/arial.ttf' if Path('C:/Windows/Fonts/arial.ttf').exists() else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
bold='C:/Windows/Fonts/arialbd.ttf' if Path('C:/Windows/Fonts/arialbd.ttf').exists() else '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
pdfmetrics.registerFont(TTFont('Arial',font))
pdfmetrics.registerFont(TTFont('ArialBold',bold))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialBold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyRU',fontName='Arial',fontSize=9.5,leading=13,spaceAfter=6))
styles.add(ParagraphStyle(name='TitleRU',fontName='ArialBold',fontSize=18,leading=22,spaceAfter=10))
styles.add(ParagraphStyle(name='HeadRU',fontName='ArialBold',fontSize=11,leading=15,spaceBefore=8,spaceAfter=5))
styles.add(ParagraphStyle(name='SmallRU',fontName='Arial',fontSize=8,leading=10,spaceAfter=4))
story=[]
def p(text,style='BodyRU'): story.append(Paragraph(text,styles[style]))
def table(rows,widths=None):
    data=[[Paragraph(str(v),styles['SmallRU']) for v in row] for row in rows]
    t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),('LINEBELOW',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,-1),(-1,-1),.5,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.append(t)
    story.append(Spacer(1,4))
def pct(v): return f'{100*v:.2f}'

names=['base','rl1','rl2','rl3']; metrics=s['metrics']
p('RL для вызова функций','TitleRU')
p('<b>Результат:</b> BFCL overall вырос с 68.93% до 75.33% (+6.40 п.п.). Лучший запуск - RL-3.')
p('Данные','HeadRU')
p(f'ToolACE: 11 300 диалогов. Использована первая пара «пользователь - ассистент». Разделение по диалогам: 90/10, seed 42. После фильтрации выбраны 2 048 примеров для обучения и 128 для валидации; пересечений нет. Исключены пустые и неразбираемые ответы, дубли и запросы длиннее 1 536 токенов. Имена функций нормализованы, вызовы разобраны через AST без исполнения.')
p(f'В обучении: {eda["train"]["with_call"]} примеров с вызовом, {eda["train"]["without_call"]} без вызова. До ограничения длины: медиана запроса {eda["prompt_tokens"]["median"]:.0f}, p90 {eda["prompt_tokens"]["p90"]:.0f} токенов; медиана ответа {eda["reference_tokens"]["median"]:.0f}; инструментов - {eda["tools"]["median"]:.0f}.')
p('Награда','HeadRU')
p('<b>Точная:</b> R = 1 за совпадение набора функций и аргументов, иначе 0. Порядок вызовов не важен. <b>Частичная:</b> R = 0.25E + Σ(0.25 + 0.5A)/max(Npred, Nref). E - точное совпадение; A - доля совпавших аргументов среди объединения ключей. Сумма берётся по однократно сопоставленным вызовам с одинаковым именем. Ошибка разметки даёт 0. Если вызов не нужен, непустой ответ без вызова даёт 1.')
p('Для weather(city="Moscow"): верный вызов -> 1; city="Paris" -> 0.25; другая функция -> 0; лишний повтор -> 0.375. Частичная награда выбрана, чтобы давать сигнал при частично верном ответе.')
p('Обучение','HeadRU')
table([['Запуск','Награда','LR','Гипотеза'],['RL-1','Точная','1e-5','Строгий критерий улучшит вызовы'],['RL-2','Частичная','1e-5','Частичный балл даст больше сигнала'],['RL-3','Частичная','3e-5','Больший LR ускорит обучение']],[55,75,50,330])
p('Qwen3.5-2B, TRL, GRPO + LoRA (r=16, α=32), без SFT. По 100 шагов с исходной модели; 4 ответа на запрос; microbatch 2, accumulation 8; β=0; group scaling; warmup 5; seed 42. Генерация при обучении: температура 0.8, до 384 токенов. Во всех запусках выбран шаг 100; LoRA объединена с базой перед оценкой.','SmallRU')
p('Результаты BFCL, %','HeadRU')
keys=['simple','multiple','parallel','parallel_multiple','irrelevance']
table([['Запуск','Overall','Simple','Multiple','Parallel','Par.+Mult.','Irrel.']]+[[n.upper(),pct(metrics[n]['overall'])]+[pct(metrics[n]['groups'][k]) for k in keys] for n in names],[60,65,65,75,70,85,50])
p('Overall - среднее пяти групп. Оценены все 350 примеров исходным evaluate.py. BFCL и токенизатор не менялись; temperature=0, max_tokens=1024, thinking выключен.','SmallRU')
p('Вывод','HeadRU')
p('Частичная награда дала +2.67 п.п. (RL-2 против RL-1), увеличение LR - ещё +3.87 п.п. (RL-3 против RL-2). В каждой паре менялась одна настройка. Наибольший измеренный эффект дал LR. На отложенном ToolACE точное совпадение: база 59.38%, RL-1 63.28%, RL-2 61.72%, RL-3 64.06%.')
p('Ограничения: один seed, 100 шагов, только первая пара диалога. Результат требует проверки повторными запусками. Команды, версии и полные ответы сохранены в README.md, results/ и runs/.','SmallRU')

story.append(PageBreak())
p('Графики и ошибки','TitleRU')
story.append(Image(str(root/'report/training_curves.png'),width=405,height=354.375))
table([['Запуск','Награда: первые -> последние 20 шагов','Группы с нулевым σ']]+[[n.upper(),f'{s["training"][n]["first_20_reward"]:.3f} -> {s["training"][n]["last_20_reward"]:.3f}',f'{s["training"][n]["zero_std_fraction"]:.1%}'] for n in names[1:]],[60,310,140])
p('При одинаковых наградах в группе advantage = 0, обучающего сигнала нет. Абсолютные награды разных формул несопоставимы. Рост награды не гарантирует рост BFCL: у RL-1 награда выросла с 0.506 до 0.519, а BFCL снизился на 0.13 п.п.')
p('Ошибки лучшей модели: 77 из 350','HeadRU')
examples=[
('irrelevance_11','Лишний вызов: на вопрос о ближайшем целом к 30 вызван get_closest_prime. Инструмент ищет простое число и не подходит.'),
('simple_225','Пропуск: модель запросила уточнение вместо psych_research.get_preference, хотя тип чтения и сравниваемые варианты уже были в запросе.'),
('simple_386','Аргумент: room_type="luxury room" вместо допустимого "luxury". Близкая формулировка не прошла строгое правило BFCL.'),
('multiple_26','Лишние вызовы: 6 вызовов game_rewards.get вместо одного. Модель придумала миссию и уровни наград.'),
('parallel_105','Смешение условий: при 350 K передан объём 1.5 вместо 2.5 м³. Количество вызовов правильное, аргумент второго вызова неверен.')]
for identifier,description in examples:
    p('<b>'+identifier+':</b> '+escape(description),'SmallRU')
SimpleDocTemplate(str(root/'report/RLHF_report.pdf'),pagesize=(595.28,841.89),leftMargin=42,rightMargin=42,topMargin=35,bottomMargin=35).build(story)
