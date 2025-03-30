import flet as ft
from settings import settings_content
from bitcoin import bitcoin_content
from finance import finance_content

def main(page: ft.Page):
    page.title = "Flet Sidebar App"
    page.window_width = 800
    page.window_height = 600
    
    content_area = ft.Column(expand=True, alignment=ft.MainAxisAlignment.START)

    def update_content(content):
        content_area.controls.clear()
        content_area.controls.append(content())
        page.update()
    
    sidebar = ft.Column(
        [
            ft.Text("Menu", size=20, weight=ft.FontWeight.BOLD),
            ft.ElevatedButton("Finance Market", on_click=lambda e: update_content(finance_content)),
            ft.ElevatedButton("Bitcoin", on_click=lambda e: update_content(bitcoin_content)),
            ft.ElevatedButton("Settings", on_click=lambda e: update_content(settings_content)),
        ],
        width=200,
        alignment=ft.MainAxisAlignment.START,
    )
    
    layout = ft.Row(
        [
            sidebar,
            ft.VerticalDivider(),
            ft.Container(content_area, expand=True, padding=20),
        ],
        expand=True,
    )
    
    page.add(layout)

ft.app(target=main)



