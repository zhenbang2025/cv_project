import gradio as gr
import torch

def check_gpu():
    is_cuda_available = torch.cuda.is_available()
    
    if is_cuda_available:
        gpu_count = torch.cuda.device_count()
        current_gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        
        result = f"""
        <h3 style='color: green;'>✅ GPU 可用！</h3>
        <p><strong>GPU 数量:</strong> {gpu_count}</p>
        <p><strong>当前 GPU 名称:</strong> {current_gpu_name}</p>
        """
    else:
        result = f"""
        <h3 style='color: red;'>❌ GPU 不可用</h3>
        <p>请检查你的 GPU 是否支持 CUDA，以及是否正确安装了 NVIDIA 驱动和 PyTorch GPU 版本。</p>
        """
    return result

with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h2>🖥️ GPU 可用性测试工具</h2>
        <p>点击下方按钮检查你的 GPU 是否可用。</p>
        """
    )
    
    with gr.Row():
        check_button = gr.Button("开始检测", variant="primary")
    
    output = gr.HTML(label="检测结果")
    
    check_button.click(check_gpu, outputs=output)

if __name__ == "__main__":
    demo.launch()