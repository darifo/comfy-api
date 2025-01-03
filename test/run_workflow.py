import torch
from app.workflows.lipstick_color_tring import LipstickColorTringFlow

if __name__ == "__main__":
    print(torch.cuda.is_available())

def main():
    flow = LipstickColorTringFlow()
    flow.run("./inputs/person.jpg", "./inputs/kouhong_cankao.jpg")

if __name__ == "__main__":
    main()