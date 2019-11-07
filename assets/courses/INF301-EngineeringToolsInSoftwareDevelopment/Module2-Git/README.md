# Mode 2 - Git. Github. Bitbucket

## 1 Hệ quản lí phiên bản

### Quản lí phiên bản phân tán

## 2 Github

## 3 Git

## 4 Bitbucket

Trong khóa học này, ta cần kết nối hệ quản lí phiên bản với nhiều công cụ khác như *Jenkins*, *Artifactory*. Vì vậy ta muốn hệ quản lí phiên bản của mình được cài đặt ở dạng *offline* (*cục bộ*). Ta có thể sử dụng phiên bản tương tự *Github* là *Gitlab*. Trong module này ta sẽ sử dụng một sản phẩm khác là ***Bitbucket***. (Tất nhiên, *Bitbucket* cũng là một sản phẩm online)

### 4.0 Tạo một tài khoản tại [Atlassian.com](my.arlassian.com)

### 4.1 Cài đặt **Bitbucket**

Để cài đặt trực tiếp trên máy, bạn có thể truy cập hướng dẫn tại [trang chủ của *Bitbucket*](https://confluence.atlassian.com/bitbucketserver/bitbucket-server-installation-guide-867338382.html)

#### 4.1.1 Cài đặt bằng *Docker*

Bạn đã biết *Docker* từ bài học trước, ta sẽ cài đặt Bitbucket bằng *Docker* trong máy ảo của mình như sau.

``` sh
anybody@anywhere:~ $ docker run -v bitbucketVolume:/var/atlassian/application-data/bitbucket --name="bitbucket" -d -p 7990:7990 -p 7999:7999 atlassian/bitbucket-server
Unable to find image 'atlassian/bitbucket-server:latest' locally
latest: Pulling from atlassian/bitbucket-server
7ddbc47eeb70: Pull complete
c1bbdc448b72: Pull complete
8c3b70e39044: Pull complete
45d437916d57: Pull complete
8737a9150f8e: Pull complete
1b5883079eff: Pull complete
60518866f62d: Pull complete
ec950f0d63d6: Pull complete
cb3e642b1f6a: Pull complete
7d276baa5e54: Pull complete
e5479665ca81: Pull complete
bc013d6a626c: Pull complete
18fe480ef142: Pull complete
Digest: sha256:45e7d31926fff661ff264bb49f724f7deb134e926b73a15b77b64e36012cd690
Status: Downloaded newer image for atlassian/bitbucket-server:latest
3b6dcca4f34c7b2551bf9dc142e3c5cc74d96ed13ace6d2a8430f2c7d0d760a7
```

#### 4.1.2 Chuyển tiếp cổng trong *Virtual Box*

Nếu bạn không dùng máy ảo, có thể truy cập ứng dụng tại `http://localhost:7990`. Nếu không, chuyển tiếp *forward* cổng này ra ngoài trong cài đặt *Virtual Box* bằng cách chọn `Settings -> Network -> Advanced -> Port Forwarding`.

<img src="assets/img/F301_2_0.png" width="800"/>

Sau đó điền luật chuyển tiếp như ở dòng *Rule 2*. (Có thể thay cổng `30779` bằng cổng khác.)

<img src="assets/img/F301_2_1.png" width="800"/>

Lưu và mở lại máy ảo. Từ đây, *Bitbucket* có thể truy cập *Bitbucket* tại `localhost:30779`. Màn hình với **Bitbucket setup** xuất hiện.

<img src="assets/img/F301_2_2.png" width="800"/>

#### 4.1.3 Tùy chỉnh trên trang web

- Ta chọn ngôn ngữ và chọn database **Internal**. Nhấn next.
- Lưu lại **Server ID** (chẳng hạn `BDSZ-RJYR-61TV-OVSD`). Nếu chưa có **license key**, chọn **I need an evaluation license**.
- Chọn "I have ...", lúc này ta được chuyển tiếp về trang của *Atlassian*. Chọn **Bitbucket (Server)**. Chọn **Your instance is: not installed yet**. Số *server ID* được điền tự động. Nếu không ta copy giá trị đã lưu ở bước trước. Sau cùng click **Generate License**.

<img src="assets/img/F301_2_3.png" width="800"/>

- License key hiện ra (bạn sẽ tìm thấy sau này khi đăng nhập lại vào *Atlassian*). Chọn **Yes** cho pop-up **Confirmation** hiện ra.
- Bây giờ license key sẽ được nhập vào trang cài đặt

<img src="assets/img/F301_2_4.png" width="800"/>

- Nhấn next và nhập các thông tin. Ta chọn **username** là `admin`.
- Chọn **Go to Bitbucket** (bỏ qua phần *Integrate with Jira*).
- Trang **Log in** hiện ra. Đăng nhập với *username* và *password* ở bước trước.

Đến đây, quá trình cài đặt hoàn tất. Ta có thể tắt và bật Bitbucket nếu cần bằng

``` sh
anybody@anywhere:~ $ docker stop bitbucket 
bitbucket
anybody@anywhere:~ $ docker start bitbucket
bitbucket
```