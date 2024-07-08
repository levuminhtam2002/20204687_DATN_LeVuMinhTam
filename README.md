# 20204687_DATN_LeVuMinhTam
Mã nguồn sử dụng để phát triển và kiểm tra các thuật toán tối ưu hóa trong bối cảnh của hệ thống MEC hỗ trợ bởi UAV
# Các thuật toán tối ưu cho hệ thống MEC hỗ trợ bởi UAV

## Tổng Quan
Repository này bao gồm mã nguồn cho các thuật toán được phát triển nhằm tối ưu hóa quá trình chuyển giao tính toán trong hệ thống MEC (Mobile Edge Computing) được hỗ trợ bởi UAV (Unmanned Aerial Vehicles). Các thuật toán này nhằm cải thiện quản lý nguồn lực và chiến lược chuyển giao, từ đó nâng cao hiệu suất của hệ thống trong nhiều điều kiện mạng khác nhau.

## Cấu Trúc Thư Mục
- **DDPG**: Thuật toán Deep Deterministic Policy Gradient cho học tăng cường trong môi trường UAV.
- **EADRL**: Thuật toán đề xuất được kết hợp bởi giải thuât tiến hóa và học tăng cường.
- **Edge_only**: Tính toán trong kịch bản chỉ vào tính toán tại edge mà không thông qua UAV.
- **GA**: Các triển khai của thuật toán di truyền để giải quyết các thách thức tối ưu hóa.
- **Local_only**: Tính toán trong kịch bản để hoạt động hoàn toàn trên các thiết bị của người dùng.

### DDPG
Thư mục này chứa mã nguồn của thuật toán học tăng cường sâu (Deep Reinforcement Learning) - DDPG (Deep Deterministic Policy Gradient). Các tệp bao gồm:
- `ddpg_algo.py`: Chứa thuật toán DDPG.
- `state_normalization.py`: Tệp này dùng để chuẩn hóa trạng thái đầu vào cho thuật toán.
- `UAV_env.py`: Môi trường mô phỏng cho UAV được sử dụng để thử nghiệm thuật toán DDPG.

### EADRL
Thư mục này chứa phiên bản thử nghiệm của thuật toán đề xuất kết hợp giải thuât tiến hóa và học tăng cường.
- `EADRL_ver1.py`: Phiên bản đầu tiên của thuật toán EADRL.
- `state_normalization.py`: Dùng để chuẩn hóa trạng thái đầu vào.
- `UAV_env.py`: Môi trường UAV cho EADRL.

### Edge_only
Thư mục này có thể chứa thuật toán chạy chỉ trên máy chủ MEC mà không xử lý dưới UE.
- `Edge_only.py`: Cài đặt tính toán cho edge computing.

### GA
Thư mục này chứa các phiên bản của thuật toán di truyền (Genetic Algorithm).
- `GA.py`: Phiên bản chính của thuật toán GA.
- `UAV_env.py` : Môi trường UAV cho thuật toán GA.

### Local_only
Thư mục này có thể chứa các thuật toán chạy hoàn toàn trên các thiết bị người dùng mà giảm tải qua máy chủ MEC.
- `Local_only.py`: Cài đặt tính toán hoàn toàn dưới thiết bị người dùng.
