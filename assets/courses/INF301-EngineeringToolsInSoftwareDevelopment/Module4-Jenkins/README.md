# Module 4 - Jenkins

## 1. Tích hợp liên tục

### 1.1 Thế nào là tích hợp?

Trong phát triển phần mềm, ***tích hơp*** (*integration*) là bước đưa phần code mà bạn đã phát triển vào phần code chung của dự án, đồng thời thực hiện kiểm tra chất lượng của phần code đó.

Lấy một ví dụ đơn giản với mô hình *git*. Giả sử dự án của bạn được lưu trữ trong một *kho* (*repository*), với nhánh chính **develop** được xem như nhánh lõi của dự án, chứa code ổn định mới nhất được phát triển. Khi một thành viên (A) phát triển và làm thay đổi code, A muốn cập nhật những thay đổi này vào nhánh lõi để các thành viên khác đều biết và cập nhật theo, anh ta gửi một *pull request* từ nhánh anh ta đã làm việc đến nhánh lõi. Các thành viên khác sẽ chấp thuận (*approve*) và sau đó trộn (*merge*) phiên bản code mới đó vào nhánh đó.

### 1.2 Thế nào là tích hợp liên tục?

Câu hỏi đặt ra: các thành viên khác *approve* như thế nào? Họ có thể tải phần code mới từ nhánh của A về, chạy các test hồi quy (*nonregression test*) hay test khác. Nếu test thành công, họ nhấn nút "approve" hay sau đó "merge". Quá trình này là một ví dụ của "tích hợp" diễn giải cho những ai đã làm quen với *git*.

Quy trình này hoàn toàn có thể tự động hóa, chẳng hạn bằng cách quy định cho một phần mềm nào đó lắng nghe mỗi lần có pull request muốn trộn code vào nhánh lõi *develop*. Phần mềm sẽ tải code mới về, chạy các test kiểm tra, sau đó approve và merge tự động vào nhánh lõi. Nếu test không thành công, người đề nghị pull request (thậm chí tất cả thành viên) sẽ nhận được email thông báo. Đây là ví dụ cho khái niệm **tích hợp liên tục** hay **tích hợp tự động** trong quy trình phát triển phần mềm. **Tích hợp liên tục** là tích hợp được thực hiện bởi một công cụ tự động, hạn chế sự can thiệp bằng tay (*manual*) từ con người.

Ngoài **tích hợp liên tục**, ta có thể nghĩ đến việc **giao sản phẩm liên tục** (*continuous delivery*) đơn giản như việc tạo ra một nhánh *release/x.y* từ nhánh *develop* đã được cập nhật (*merged*), như là sự ghi dấu cho việc phiên bản *x.y* của phần mềm đã ra đời. Thậm chí, ta có thể đẩy (*push*) phiên bản này lên môi trường kiểm tra (*test*, *qualification*, *staging*) một cách tự động để thay thế cho phiên bản cũ trước đó.

### 1.3 Jenkins

**Jenkins** được tạo ra với mục đích chính để thực hiện tích hợp liên tục và giao sản phẩm liên tục.

Ngoài ra, nhiều dự án còn sử dụng Jenkins như một tiện ích thực hiện các công việc (*job*) thường dành cho các *script*, đặc biệt là những việc được lên kế hoạch (*scheduled*) thực hiện theo chu kì một cách tự động.

Nhiều thông tin liên quan tại [jenkins.io](https://jenkins.io/).

*Jenkins* không phải là công cụ duy nhất được sử dụng trong tích hợp liên tục. Ta có thể nghĩ đến mô hình với các công cụ liên quan:

- Công cụ bảo trì code: *Git*/*Bitbucket*, ...
- Công cụ kiểm tra: *JUnit* hoặc ngay bản thân ngôn ngữ bạn dùng để viết code, ...
- Công cụ xuất bản: *Artifactory*, ... (dùng khi phiên bản cần giao hàng chứa nhiều định dạng hay chứa định dạng phức tạp)

*Jenkins* dễ dàng liên kết với nhiều công cụ trong chuối này thông qua các *giác cắm* (*plugin*) của nó.

## 2. Cài đặt Jenkins

Như đã nói ở các module trước, để dễ dàng minh họa và không phụ thuộc vào môi trường bạn đang làm việc, ta sử dụng Docker để minh họa cách cài đặt Jenkins.

Để xem hướng dẫn cài đặt trực tiếp, bạn có thể truy cập [trang ví dụ của jenkins.io](https://jenkins.io/doc/pipeline/tour/getting-started/).

### 2.1 Sao chép (*clone*) kho github cho docker jenkins

``` sh
anybody@anywhere:~/workspace $ git clone https://github.com/jenkinsci/docker.git
Cloning into 'docker'...
remote: Enumerating objects: 2639, done.
remote: Total 2639 (delta 0), reused 0 (delta 0), pack-reused 2639
Receiving objects: 100% (2639/2639), 606.23 KiB | 1.35 MiB/s, done.
Resolving deltas: 100% (1414/1414), done.
```

### 2.2 Chuyển đến thư mục vừa tải

```sh
anybody@anywhere:~/workspace $ mv docker jenkins-docker

# Đổi tên để tiện làm việc
anybody@anywhere:~/workspace $ cd jenkins-docker/

# Kiểm tra bên trong thư mục
anybody@anywhere:~/workspace/jenkins-docker master ± ls
CHANGELOG.md        Dockerfile-jdk11    jenkins-support          publish.sh
CONTRIBUTING.md     Dockerfile-slim     LICENSE.txt              README.md
docker-compose.yml  HACKING.adoc        Makefile                 tests
Dockerfile          install-plugins.sh  multiarch                tini_pub.gpg
Dockerfile-alpine   Jenkinsfile         plugins.sh               tini-shim.sh
Dockerfile-centos   jenkins.sh          publish-experimental.sh  tools
```

### 2.3 Chạy container docker

```sh
# Chạy docker, mở cổng 8080 cho ứng dụng web và 50000 cho ứng dụng khác
anybody@anywhere:~/workspace/jenkins-docker master ± docker run -p 8080:8080 -p 50000:50000 jenkins/jenkins:lts
Running from: /usr/share/jenkins/jenkins.war

...

Jenkins initial setup is required. An admin user has been created and a password generated.
Please use the following password to proceed to installation:
# QUAN TRỌNG: Copy password này để cài đặt trên giao diện
2cbfe5490ea14622abfb8c941358ee34

This may also be found at: /var/jenkins_home/secrets/initialAdminPassword

*************************************************************
*************************************************************
*************************************************************

2019-11-04 16:40:16.615+0000 [id=46] INFO    hudson.model.UpdateSite#updateData: Obtained the latest update center data file for UpdateSource default
2019-11-04 16:40:17.292+0000 [id=30]    INFO    hudson.model.UpdateSite#updateData: Obtained the latest update center data file for UpdateSource default
2019-11-04 16:40:17.694+0000 [id=30]    INFO    jenkins.InitReactorRunner$1#onAttained: Completed initialization

# Đây là dòng cho thấy Jenkins đã được cài đặt thành công
2019-11-04 16:40:17.720+0000 [id=20]    INFO    hudson.WebAppMain$3#run: Jenkins is fully up and running
2019-11-04 16:40:18.157+0000 [id=46]    INFO    h.m.DownloadService$Downloadable#load: Obtained the updated data file for hudson.tasks.Maven.MavenInstaller
2019-11-04 16:40:18.158+0000 [id=46]    INFO    hudson.util.Retrier#start: Performed the action check updates server successfully at the attempt #1
2019-11-04 16:40:18.161+0000 [id=46]    INFO    hudson.model.AsyncPeriodicWork$1#run: Finished Download metadata. 8,706 ms
```

### 2.4 Sử dụng cài đặt Virtual Box chuyển tiếp (*forward*) cổng máy ảo sang máy chính

Ví dụ, chuyển tiếp cổng 8080 từ máy ảo (máy khách, *guest*) sang 30880 của máy chính (máy chủ, *host*) bằng cách chọn: `Settings -> Network -> Advanced -> Port Forwarding`.

<img src="assets/img/F301_4_1.png" width="800"/>

Sau đó điền luật chuyển tiếp như sau:

<img src="assets/img/F301_4_2.png" width="800"/>

Bạn có thể chọn tên luật tùy ý. IP máy ảo và máy chính có thể dùng `"0.0.0.0"`. Bạn cũng có thể dùng cổng khác của máy chủ thay vì 30880.

### 2.5 Cài đặt với giao diện web trên máy chính

Sau bước 2.4, lúc này bạn có thể truy cập giao diện web của Jenkins tại `http://localhost:30880`. Thay `30880` nếu bạn sử dụng cổng khác.

Điền password đã lưu từ terminal vào khung trống

<img src="assets/img/F301_4_3.png" width="800"/>

Ở bước tiếp theo: chọn **`Install suggested plugins`**. Ứng dụng sẽ được cài đặt, bạn đồng thời sẽ nhìn thấy các dòng logs tại terminal trong máy ảo nơi bạn đã gõ lệnh chạy *Jenkins* với *docker*. Đồng thời logs và quá trình cài đặt cũng hiện ra trên giao diện web.

<img src="assets/img/F301_4_19.png" width="800"/>

### 2.6 Khai báo quản trị viên

Sau bước 2.5, Jenkins sẽ đưa bạn đến trang dành cho khai báo quản trị viên. Chẳng hạn, ta khai báo quản trị viên `admin`.

<img src="assets/img/F301_4_5.png" width="800"/>

Nháy vào `Save and Continue`. Trang ***Instance Configuration*** mở ra với URL của Jenkins được định sẵn là *`http://localhost:30880/`*. Đến đây việc cài đặt *Jenkins* hoàn tất. Bạn có giao diện như sau:

<img src="assets/img/F301_4_6.png" width="800"/>

### 2.7 Tắt, bật Jenkins

Khi tắt *Jenkins* với `Ctrl + C` trong terminal, container bị phá hủy. Ta có thể bật lại Jenkins bằng `docker start`.

``` sh
# Tìm id của container
anybody@anywhere:~/workspace/jenkins-docker master 127 ± docker ps -a
CONTAINER ID        IMAGE                 COMMAND                  CREATED             STATUS                            PORTS               NAMES
1fbcd6c56f3b        jenkins/jenkins:lts   "/sbin/tini -- /usr/…"   14 minutes ago      Exited (130) About a minute ago                       practical_mayer

# Bật lại container
anybody@anywhere:~/workspace/jenkins-docker master ± docker start 1fbcd6c56f3b
1fbcd6c56f3b
```

Nếu muốn tắt container lần nữa:

``` sh
anybody@anywhere:~/workspace/jenkins-docker master ± docker stop 1fbcd6c56f3b
```

Nếu muốn xóa container

``` sh
anybody@anywhere:~/workspace/jenkins-docker master ± docker rm 1fbcd6c56f3b
```

Tuy nhiên, nếu xóa container, toàn bộ dữ liệu làm việc như các *công việc* (*job*), *cấu hình* (*configuration*) cũng mất. Phần tiếp theo cho biết cách *dựng* (*mount*) một *thể tích* (*volume*) chứa dữ liệu trong máy ảo để giữ được dữ liệu ngay khi container đã bị xóa.

### 2.8 Bật Jenkins và dựng dữ liệu

Mọi dữ liệu (*công việc*, *cấu hình*) của Jenkins được lưu tại `/var/jenkins_home` trong container. Để dựng *thể tích* (*volume*) chứa dữ liệu `jenkins_home` trong máy ảo đồng bộ với `/var/jenkins_home` trong container, ta dùng lệnh sau:

``` sh
anybody@anywhere:~/workspace/jenkins-docker master ± docker run -p 8080:8080 -p 50000:50000 -v jenkins_home:/  jenkins/jenkins:lts
Running from: /usr/share/jenkins/jenkins.war
webroot: EnvVars.masterEnvVars.get("JENKINS_HOME")
...
Jenkins initial setup is required. An admin user has been created and a password generated.
Please use the following password to proceed to installation:

1917fc7092de49328326056b71155ad8

...
2019-11-04 17:35:27.694+0000 [id=20]    INFO    hudson.WebAppMain$3#run: Jenkins is fully up and running
```

Để kiểm tra volume

``` sh
anybody@anywhere:~/workspace/jenkins-docker master ± docker volume ls
DRIVER              VOLUME NAME
...
local               jenkins_home
```

## 3. Các ví dụ:

### 3.1 Sử dụng Jenkins như một *mẫu* (*form*) để thực hiện một công việc

#### 3.1.1 Kịch bản

Bạn muốn tìm thông tin về tọa độ của một địa chỉ. Việc tìm thông tin được thực hiện thông qua API

``` txt
https://api-adresse.data.gouv.fr/search/
```

Chẳng hạn, ta gửi yêu cầu trực tiếp từ terminal

``` bash
anybody@anywhere:~ $ curl "https://api-adresse.data.gouv.fr/search/?q=14+avenue+de+la+Rostagne"
{"type": "FeatureCollection", "version": "draft", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [7.118747, 43.571669]}, "properties": {"label": "14 Avenue de la Rostagne 06160 Antibes", "score": 0.8812350699438987, "housenumber": "14", "id": "06004_3020_00014", "type": "housenumber", "name": "14 Avenue de la Rostagne", "postcode": "06160", "citycode": "06004", "x": 1032680.91, "y": 6283412.48, "city": "Antibes", "context": "06, Alpes-Maritimes, Provence-Alpes-C\u00f4te d'Azur", "importance": 0.6935857693828857, "street": "Avenue de la Rostagne"}}], "attribution": "BAN", "licence": "ODbL 1.0", "query": "14 avenue de la Rostagne", "limit": 5}
```

Thay vì thiết kế một mẫu trên web để thực hiện hoặc dùng `curl` trực tiếp từ console, ta có thể dùng Jenkins

#### 3.1.2 Viết job

Nếu Jenkins đang tắt, mở lại bằng

``` bash
docker start 1fbcd6c56f3b
```

(`1fbcd6c56f3b` là id của container). Đăng nhập bằng tài khoản quản trị viên đã thiết lập trước đó nếu cần.

Thao tác tạo *công việc* (*job*)

- Click **New item** ở menu trái.
- Trong khung **Enter an item name**, điền tên *job*, chẳng hạn `RequestAddress`.
- Chọn kiểu của *job*. Ta sẽ chọn **Pipeline** để biểu diễn công việc thành một chuỗi công đoạn nhỏ. *Pipeline* mang nghĩa đường ống cho ta hình dung một chuỗi công việc như dòng nước chảy.
- Click **OK**. Lúc này bạn có:

<img src="assets/img/F301_4_7.png" width="800"/>

Đây là *cấu hình* (*configuration*) của công việc. Bạn có thể tìm lại nó bằng cách chọn **Configure** ở menu trái. Nhiều chức năng có thể được tùy chỉnh. Trong kịch bản, ta cần một trường (*field*) cho phép nhập địa chỉ cần tìm kiếm. Do vậy:

- Click **This project is parameterized.**
- Chọn **Add Parameter** -> **String Parameter**, đặt tên (**Name**) cho nó là "ADDRESS"

<img src="assets/img/F301_4_10.png" width="800"/>

- Trong phần **Pipeline** -> **Definition**, bạn có thể thử viết đoạn mã bằng cách tham khảo ***try sample Pipeline*** hoặc dùng code tại [src/RequestAddress](src/RequestAddress). Ngôn ngữ được dùng trong mỗi *stage* là **groovy**. Lưu ý: Nhìn chung, bạn có thể viết mã ở các ngôn ngữ khác (*bash*, *python*), đặt trong kho quản lí phiên bản (như *github*) và chạy nó bằng lệnh `sh myscript.sh` hay `python myscript.python`. Phần này sẽ được giới thiệu sau. Do đó, bạn không cần quá quan tâm về *groovy*.

<img src="assets/img/F301_4_11.png" width="800"/>

- Bỏ chọn **Use Groovy Sandbox**. Sandbox là công cụ để hạn chế script thực hiện một số thao tác do vấn đề bảo mật. Hiện ta đang thực hiện một thao tác thử đơn giản nên việc bỏ chọn không ảnh hưởng nhiều.

- Nhấn **Save**

#### 3.1.3 Chạy job

- Ở menu trái, chọn **Build with Parameters**
- Điền một địa chỉ nào đó ở Pháp trong khung

<img src="assets/img/F301_4_12.png" width="800"/>

- Ấn **Build** (bạn có thể thực hiện nhiều lần)

- Nếu job thất bại, ta có trạng thái như #1 của hình dưới. Nếu job thành công, trạng thái như #2.

<img src="assets/img/F301_4_13.png" width="800"/>

- Click vào **#2** (hoặc một lần *Buiid* thành công), rồi **Console Output** ta nhìn thấy *logs* của job.

<img src="assets/img/F301_4_14.png" width="800"/>

 Từ logs, ta thấy job đã thực hiện 3 chặng giống như trong mô tả code trong [src/RequestAddress](src/RequestAddress). Ở chặng 1, job thực hiện việc tải dữ liệu từ API. Ở chặng 2, job tách dữ liệu và tìm các thông tin cần thiết (kinh vĩ độ, mã bưu điện). Ở chặng 3, job hiển thị kết quả.

- Để thấy rõ hơn các bước, ta có thể chọn **Pipeline Steps** ở menu trái.

<img src="assets/img/F301_4_15.png" width="800"/>

Bạn có thể chơi với các mục khác trên thanh menu trái, đặc biệt là thay đổi các cấu hình trong **Configure**.

Nhắc lại rằng bạn có thể tắt và bật lại docker container để tìm lại trạng thái đã làm việc.

#### 3.1.4 Nhập (*import*) code của job từ xa

Ta có thể *import* code của job từ xa (chẳng hạn, *github*) thay vì viết trực tiếp vào ô *Script* trong mục *Pipeline* của công việc.

Để minh họa, giả sử code của job ở phần 3.1.3 đã được đưa lên [github](https://github.com/riduan91/Training/blob/master/INF301-EngineeringToolsInSoftwareDevelopment/Module4-Jenkins/src/RequestAddress.groovy)

Bây giờ, ta tạo một *project/job* mới bằng cách (click vào biểu tượng *Jenkins* để về trang chủ nếu cần rồi) click vào **Create new item**. Đặt tên mới, chẳng hạn `RequestAddressGithub`, cho project.

Tiếp tục quy trình:

- Chọn **This project is parameterized**, Thêm biến `ADDRESS`. (xem 3.1.2)
- Ở mục **Pipeline**, chọn `Pipeline script from SCM`.
- Ở **SCM**, chọn `Git`.
- Ở **Repository URL**, diền đường dẫn của kho *github*. Thêm tài khoản (*credentials*) của Github của bạn. Chọn branch trên *Github*.
- Ở **Script Path**, điền đường dẫn đến file chứa mã.
- **Save**

<img src="assets/img/F301_4_22.png" width="800"/>

- **Build** *job* như ví dụ trước. Chú ý rằng lần này build sẽ thất bại vì ta buộc phải dùng *Groovy Sandbox*. Trong **console log**, ta thấy lỗi như sau:

<img src="assets/img/F301_4_17.png" width="800"/>

 Click vào **Administrators can decide whether to approve or reject this signature.** và **approve** *signature : new groovy.json.JsonSlurperClassic*, ta thấy dòng này xuất hiện trong khung:

<img src="assets/img/F301_4_18.png" width="800"/>

- Bây giờ ta **build** lại lần nữa (ta có thể gặp lại lỗi và cần *approve* lần nữa cho một hàm tương tự). Sau không quá 3 lần build, ta thấy *job* thành công. So với *logs* ở 3.1.3, bây giờ logs có thêm phần tương tác với git:

``` sh
Checking out git https://github.com/riduan91/Training.git into /var/jenkins_home/workspace/RequestAddressGithub@script to read INF301-EngineeringToolsInSoftwareDevelopment/Module4-Jenkins/src/RequestAddress.groovy
using credential Github
 > git rev-parse --is-inside-work-tree # timeout=10
Fetching changes from the remote Git repository
 > git config remote.origin.url https://github.com/riduan91/Training.git # timeout=10
Fetching upstream changes from https://github.com/riduan91/Training.git
 > git --version # timeout=10
using GIT_ASKPASS to set credentials
 > git fetch --tags --progress -- https://github.com/riduan91/Training.git +refs/heads/*:refs/remotes/origin/* # timeout=10
 > git rev-parse refs/remotes/origin/master^{commit} # timeout=10
 > git rev-parse refs/remotes/origin/origin/master^{commit} # timeout=10
Checking out Revision 3e774765aa885e5bc86526d18f7ec1705827f77c (refs/remotes/origin/master)
 > git config core.sparsecheckout # timeout=10
 > git checkout -f 3e774765aa885e5bc86526d18f7ec1705827f77c # timeout=10
Commit message: "INF 301 - Module 4 - Jenkins example (cont'd)"
 > git rev-list --no-walk 3e774765aa885e5bc86526d18f7ec1705827f77c # timeout=10
Running in Durability level: MAX_SURVIVABILITY
[Pipeline] Start of Pipeline
[Pipeline] node
Running on Jenkins in /var/jenkins_home/workspace/RequestAddressGithub
[Pipeline] {
...
```

Đến đây việc lưu trữ code của *job* từ *Github* đã thành công. Nhớ rằng bạn có thể tắt và bật lại docker container để tìm lại trạng thái đã làm việc.

## 4. Ví dụ 2: Kiểm tra tự động với Jenkins

## 5. Ví dụ 3: Trộn tự động với Jenkins

## 6. Ví dụ 4: Giao hàng tự động với Jenkins

## Nguồn tư liệu

1. [github.com/jenkinsci/docker](https://github.com/jenkinsci/docker)
2. [jenkins.io](https://jenkins.io)
