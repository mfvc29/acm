from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import os

def generar_pdf(precio_estimado, precio_estimado_dolares, tipo_cambio, zona, municipio, tipo_inmueble, propiedades_similares, precio_max, diferencia):
    # Configuración de Jinja2 para cargar el template
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('template.html')

    # Datos para renderizar el template
    context = {
        'zona': zona,
        'municipio': municipio,
        'tipo_inmueble': tipo_inmueble,
        'precio_estimado': f'{precio_estimado:,.2f}',
        'precio_estimado_dolares': f'{precio_estimado_dolares:,.2f}',
        'tipo_cambio': f'{tipo_cambio:,.2f}',
        'precio_max': f'{precio_max:,.2f}',
        'diferencia': f'{diferencia:,.2f}',
        'propiedades_similares': propiedades_similares
    }

    # Renderizar el HTML con los datos
    html_content = template.render(context)

    # Generar el PDF usando WeasyPrint
    html = HTML(string=html_content)
    pdf_path = "resultado_propiedad.pdf"
    html.write_pdf(pdf_path)

    return pdf_path

# Ejemplo de datos (puedes adaptarlos)
precio_estimado = 941737.24
precio_estimado_dolares = 247825.59
tipo_cambio = 3.80
zona = "Punta Hermosa"
municipio = "Lima Sur"
tipo_inmueble = "Casa"
precio_max = 1424664.00
diferencia = precio_max - precio_estimado

# Datos de propiedades similares
propiedades_similares = [
    {'area_total': 155, 'dormitorios': 3, 'banos': 3, 'estacionamiento': 2, 'precio_venta': 1368000},
    {'area_total': 157, 'dormitorios': 3, 'banos': 5, 'estacionamiento': 2, 'precio_venta': 712125},
    {'area_total': 160, 'dormitorios': 3, 'banos': 1, 'estacionamiento': 2, 'precio_venta': 518000},
]

# Generar PDF
pdf_path = generar_pdf(precio_estimado, precio_estimado_dolares, tipo_cambio, zona, municipio, tipo_inmueble, propiedades_similares, precio_max, diferencia)
print(f"PDF generado: {pdf_path}")
