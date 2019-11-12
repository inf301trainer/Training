# Module 2 - Git. Github. Bitbucket

## 1 Hệ quản lí phiên bản

### 1.1 Quản lí phiên bản

Tưởng tượng bạn làm việc nhiều người trong cùng một dự án, trên cùng một tập hợp các file code và dữ liệu. Tất cả các thành viên đều thay đổi thường xuyên code và mỗi người, sau một thời gian, cần dùng phiên bản code mới được cập nhật bởi các thành viên code. Sẽ là tai họa nếu không có một sự quản lí tập trung các phiên bản code mà mỗi người làm việc.

Bạn có thể liên tưởng đến *Google Drive*/*Google Docs* khi mọi thành viên làm việc trên cùng một tài liệu và xem đây là một giải pháp. Nghĩa là:

- Các thành viên làm việc trực tiếp trên một file
- Các thành viên đều phải có mạng để truy cập đến file đó.

Giải pháp này dẫn đến vấn đề: các va chạm sẽ diễn ra thường xuyên, đặc biệt khi các thành viên sửa chữa tài liệu tại cùng một dòng. Tình huống còn nghiêm trọng hơn khi đối tượng ta làm việc là code chứ không phải tài liệu. Nếu bạn dùng phiên bản tại thời điểm *A* nào đó khi các thành viên khác chưa viết xong một câu lệnh, hiển nhiên code sẽ lỗi và không thực hiện được.

Đấy là lí do ta cần một **hệ quản lí phiên bản** (**Version Control System** hay **VCS**). Code của dự án sẽ được lưu trữ trong một *thư mục* hay *kho* (với *git* sẽ tìm hiểu ở phần tiếp theo, thuật ngữ tiếng Anh là **repository**). Mỗi thành viên sẽ tải về một phiên bản, làm việc trên đó để đám bảo tính cục bộ (*local*) và độc lập với các thành viên khác. Chỉ khi việc cập nhật code trở nên hoàn chỉnh, thành viên mới cập nhật code vào kho chính online.

**Hệ quản lí phiên bản** giúp quản lí code và lịch sử thay đổi, từ đó bạn có thể quay lại một phiên bản trước đó, hoặc dùng một phiên bản do người khác cập nhật nếu cần.

### 1.2 Hệ quản lí phiên bản tập trung

Hình dưới đây mô tả một **hệ quản lí phiên bản tập trung** (**centralized version control system**).

<img src="assets/img/F301_2_5.png" width="300"/>

(Nguồn: [homes.cs.washington.edu](http://homes.cs.washington.edu))

Theo đó, có một *kho* (**repository**) online (vẽ ở tầng trên cùng). Các thành viên sẽ tải kho này về thành các thư mục cục bộ (*working copy* trong hình). Hành động tải này được thể hiện bằng mũi tên *update* trong hình. Sau khi sửa chữa code, thành viên cập nhật phiên bản thư mục cục bộ của mình lên kho online. Hành động này được thể hiện bằng mũi tên *commit*. Hệ quản lí phiên bản tập trung hay được sử dụng nhất là [Subversion](https://subversion.apache.org/). Nhìn chung, nếu có hai thành viên cùng sửa chữa từ một phiên bản code online (tạm gọi là *A*) và *commit* 2 phiên bản mới (tạm gọi là *B1* và *B2*), hệ cho phép trộn lẫn hai phiên bản này thành một phiên bản chung (*B*) gồm cả các thay đổi của hai người, nếu chúng không va chạm (*conflict*) với nhau.

### 1.3 Hệ quản lí phiên bản phân tán

Với một **hệ quản lí phiên bản phân tán** (**distributed version control system**):

<img src="assets/img/F301_2_6.png" width="300"/>

(Nguồn: [homes.cs.washington.edu](http://homes.cs.washington.edu))

Ta thấy có thêm một tầng *online* nữa nằm giữa kho chính và các thư mục cục bộ. Đó chính là các phiên bản *online* hoặc *offline* khác, ban đầu được sao chép từ kho chính, được biểu diễn bằng *Repository* trong tầng giữa của hình vẽ. Trong *Github* các kho này có thể được gọi bằng khái niệm **nhánh** (**branch**) nếu phiên bản được sao chép nằm ở cùng tài khoản với kho gốc; hoặc được gọi là **bản sao**/**bản chĩa** (**fork**) nếu nó nằm ở nằm ở tài khoản của người dùng. Dù gọi là gì, chúng vẫn là một phiên bản *online* của kho chính. Các thành viên *update* và *commit* trên các phiên bản (*fork*, *branch*) này. Việc cập nhật code lên kho chính thức được thực hiện bằng các thao tác riêng (mô tả bằng **pull** và **push** trong hình vẽ).

#### Chú ý:

Hình vẽ trên mô tả đơn giản khái niệm *hệ quản lí phiên bản phân tán*, tuy vậy nó chưa đầy đủ với **git**. Trên thực tế, ngoài các động từ *pull*, *push*, ta còn có *clone*, *fork*, *create branch*, *pull request*, *merge* với các chức năng khác nhau. Ta sẽ tìm hiểu kĩ hơn trong các phần tiếp theo.

### 1.4 Ưu điểm của *hệ quản lí phiên bản phân tán* so với *hệ tập trung*

Bài báo tại [itviec.com](https://itviec.com/blog/git-la-gi/) nêu quan điểm của những người có kinh nghiệm về việc tại sao hệ phân tán có ưu thế hơn (*Sắp xếp công việc tốt hơn*, *Linh hoạt khi làm nhiều công việc*, *Tự tin thể hiện ý tưởng mới etc.*). Các lợi ích này đều bắt nguồn từ tính độc lập lẫn nhau giữa các nhánh.

## 2 Git

Mô tả ở phần 1 là mô hình cơ bản của *hệ quản lí phiên bản phân tán*. **git** là *framework* (*cơ cấu*) cho hệ quản lí loại này, với mô hình cụ thể hơn như sau:

<img src="assets/img/F301_2_7.png" width="800"/>

Ta sẽ giải thích các tầng và các khái niệm trong mô hình.

Cơ cấu được tổ chức bằng cách ở hai cấp *remote* (*online*, với code được lưu trong một sản phẩm online như **Github**, **Bitbucket**) và  hay *local* (hay *offline*, với code được lưu trong máy người dùng). Hai tầng trên cùng trong hình là tầng *remote*, hai tầng dưới là *local*.

Phần này chỉ mô tả sơ lược các khái niệm. Về ví dụ thực hành, ta tìm hiểu ở phần 3 và 4 sau.

## 2.1 Các khái niệm

Trong hình vẽ,

- **Remote repository** là một *thư mục*/*kho* online chứa code của dự án. Các hộp màu xanh lá cây và xanh dương là các *remote repository*.

- **Main repository** (đôi khi được gọi là *upstream*) là thư mục/kho code chính của dự án. Trong thư mục này có một nhánh (**branch**), thường được đặt tên là **develop**, là nhánh chính của dự án. Trong hình vẽ, hộp màu xanh lá với branch *develop* là *main repository* và *branch* chính của dự án.

- **(Remote) Branch** là một copy của một *repository*, có thể mang code giống hay khác branch chính của *main repository*. Trong hình vẽ, hộp xanh dương bên trái mô tả một *branch* của *main repository*.

- **Forked repository** là thư mục/kho code được copy từ *main repository* về tài khoản của một người dùng khác (trên *Github* hay sản phẩm quản lí phiên bản online). *Forked repository* chứa các *branch* từ *main repository* và người dùng cũng có thể tạo các *branch* khác. Trong hình vẽ, hộp xanh dương bên phải mô tả một *branch* của *main

Như vậy, ở cấp online, một phiên bản code tại một thời điểm được đồng nhất với một nhánh (*branch*) của một kho (*repository*) tại thời điểm đó.

Tại cùng thời điểm có thể tồn tại nhiều phiên bản code khác nhau (ứng với nhiều người dùng khác nhau, tại các phần dự án khác nhau).

Tiếp theo,

- **Local repository**, **local branch** là cấu trúc git copy từ phiên bản online (*remote repository*, *remote branch*) về máy tính người dùng, chứa cấu trúc git. Trong hình vẽ, nó ứng với tầng thứ ba (các hộp màu da cam). Người ta cũng gọi tầng này là *staging environment*.

- **Working directory** là phiên bản code được lưu trên máy người dùng ứng với branch mà người dùng đang làm việc ở thời điểm hiện tại. Nó được lưu trong một thư mục trong máy tính cá nhân người dùng.

Trong hình vẽ cũng đưa ra 3 phương án tiếp cận với *main repository*. Ở nhánh giữa, người dùng tải trực tiếp *main repository* về *local repository*. Ở nhánh bên trái, người dùng tạo một nhánh (*branch*) online trong chính *main repository* và tải phiên bản đó về *local repository*. Ở nhánh bên phải, người dùng copy *main repository* về tải khoản riêng, thành *forked repository* và tải phiên bản đó về *local repository*.

Ở phần tiếp theo, ta minh họa tương tác giữa working repository trên máy với *Github*.

## 3 Github

### 3.1 Tạo tài khoản Github

Nếu chưa có tài khoản, bạn tạo tài khoản tại [github.com](http://github.com) (Chọn **Sign up** -> Điền *username*, *email* và chọn *password*. Chọn **Free** nếu được hỏi *Pick the plan that's right for you*. Kiểm tra email và kích vào **Verify email address** để kích hoạt tài khoản.)

Ví dụ, tài khoản để sử dụng trong tutorial này là `inf301trainer`.

### 3.2 Minh họa nhánh giữa của hình: làm việc trực tiếp với nhánh chính của *remote repository*

#### 3.2.1 Tạo (*create*) repository

Ta minh họa cách tạo ra *main repository*.

<img src="assets/img/F301_2_9.png" width="300"/>

- Tại tài khoản của bạn ở [github.com](https://github.com), click vào kí hiệu **+** và **New repository**.

- Đặt tên thư mục và click **Create repository**.

<img src="assets/img/F301_2_10.png" width="800"/>

- Ở trang tiếp theo, *Github* hướng dẫn cách set up thư mục cục bộ. Ta bỏ qua phần này, sẽ được hướng dẫn ở phần tiếp theo.

#### 3.2.2 Cài đặt *git* trên máy cục bộ

Bạn có thể cài đặt **git** trên môi trường thích hợp theo hướng dẫn tại [https://www.atlassian.com/git/tutorials/install-git](https://www.atlassian.com/git/tutorials/install-git), hoặc cài đặt một *ova* trong *Virtual box* chứa sẵn *git*.

#### 3.2.3 *git clone*: Tải *remote repository* về *local repository* lần đầu

<img src="assets/img/F301_2_14.png" width="300"/>

**Clone** là động tác tải *remote repository* (có thể là *main repository* hoặc *forked repository*) về *local repository* để tạo lập *local repository* lần đầu tiên. Khi đó, *working repository* cũng được tạo tự động với nhánh ban đầu (trong *Github* là **master** thay cho **develop**.)

Để minh họa:

``` sh
# Đến 1 thư mục nơi bạn có thể lưu code
anybody@anywhere:~/workspace $ git clone https://github.com/inf301trainer/inf301.git
Cloning into 'inf301'...
warning: You appear to have cloned an empty repository.
```

Hiển nhiên hiện tại thư mục chưa có gì nên ta nhận được warning như trên.

#### 3.2.4 *git commit*: Cập nhật code từ *working directory* đến *local (git) repository*

<img src="assets/img/F301_2_15.png" width="300"/>

``` sh
# Di chuyển đến thư mục
anybody@anywhere:~/workspace $ cd inf301/
# Tạo 1 hay nhiều file code
anybody@anywhere:~/workspace/inf301 master ± echo 'print("Hello World!")' > helllo_world.py
# Kiểm tra code đã được tạo
anybody@anywhere:~/workspace/inf301 master* ± ls
helllo_world.py
# Thêm tất cả các file vào local git repository. git add có thể thực hiện nhiều lần trước khi git commit
anybody@anywhere:~/workspace/inf301 master* ± git add -A
# git commit: Tập hợp tất cả các lần git add (thay đổi code) thành 1 đóng góp duy nhất. Option -m là message ứng với đóng góp
anybody@anywhere:~/workspace/inf301 master ± git commit -m "Add 'hello_world' script"
# Đôi khi lỗi xảy ra vì bạn chưa khai báo tài khoản
*** Please tell me who you are.

Run

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

to set your account s default identity.
Omit --global to set the identity only in this repository.

fatal: unable to auto-detect email address (got 'anybody@anywhere.(none)')
# Khai báo tài khoản bằng git config --global nếu bạn muốn khai báo cho tất cả working directory, git config --local nếu bạn muốn khai báo cho chỉ 1 working directory.
anybody@anywhere:~/workspace/inf301 master 128 ± git config --local user.name inf301trainer
anybody@anywhere:~/workspace/inf301 master ± git config --local user.email master@gmail.com
# Thực hiện git commit
anybody@anywhere:~/workspace/inf301 master ± git commit -m "Add 'hello_world' script"
[master (root-commit) 14541b9] Add 'hello_world' script
 1 file changed, 1 insertion(+)
 create mode 100644 helllo_world.py
# Đến đây ta phát hiện ra tên file bị lỗi, ta có thể thay đổi tên file
anybody@anywhere:~/workspace/inf301 master ± mv helllo_world.py hello_world.py
anybody@anywhere:~/workspace/inf301 master(+0/-1)* 1 ± git add -A
anybody@anywhere:~/workspace/inf301 master(+0/-0) ± git commit -m "Rename hello_world"
[master 483530b] Rename hello_world
 1 file changed, 0 insertions(+), 0 deletions(-)
 rename helllo_world.py => hello_world.py (100%)
```

Đến đây việc đóng góp thay đổi (*commit*) hoàn tất.

#### 3.2.5 *git push*: Cập nhật code từ *local (git) repository* đến *remote repository*

<img src="assets/img/F301_2_16.png" width="300"/>

``` sh
# Đẩy code lên remote repository (origin), nhánh master
anybody@anywhere:~/workspace/inf301 master ± git push origin master
Username for 'https://github.com': inf301trainer
Password for 'https://inf301trainer@github.com':
Counting objects: 5, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (2/2), done.
Writing objects: 100% (5/5), 470 bytes | 470.00 KiB/s, done.
Total 5 (delta 0), reused 0 (delta 0)
To https://github.com/inf301trainer/inf301.git
 * [new branch]      master -> master
```

Để kiểm tra `origin` là gì

``` sh
anybody@anywhere:~/workspace/inf301 master 129 ± git remote -v
origin  https://github.com/inf301trainer/inf301.git (fetch)
origin  https://github.com/inf301trainer/inf301.git (push)
```

Lúc này, trên tài khoản Github, ta thấy code đã được cập nhật (`hello_world.py`)

<img src="assets/img/F301_2_17.png" width="600"/>

Nếu click vào **2 commits**, ta có thể thấy danh sách các commits cùng các message của chúng. Đây là cơ sở giúp ta tìm lại các phiên bản code trước đó.

<img src="assets/img/F301_2_18.png" width="600"/>

Mô hình này thích hợp khi bạn là người làm việc duy nhất với *remote repository*. Để tiếp tục làm việc và đẩy code lên *remote repository*, ta thực hiện luân phiên `git add`, `git commit`, `git push`.

### 3.3 Minh họa nhánh trái của hình: làm việc với nhánh phụ của *remote repository* và trộn vào nhánh chính

Khi có 2 người trở lên cùng làm việc với cùng một *remote repository*, sẽ an toàn hơn khi ta dùng các nhánh riêng.

#### 3.3.1 Tạo nhánh (*create branch*)

<img src="assets/img/F301_2_21.png" width="600"/>

Giả sử ta muốn tạo 2 nhánh *A*, *B* cho 2 tính năng của dự án, được phát triển bởi 2 nhóm lập trình viên.

Muốn vậy,

- Từ *remote repository* trong github.com, click tại **master**, tạo nhánh (*branch*) **A** từ (*from*) **master**. Làm tương tự với **B**.

  <img src="assets/img/F301_2_20.png" width="600"/>

- Lúc này, ta có danh sách các nhánh:

  <img src="assets/img/F301_2_22.png" width="200"/>

#### 3.3.2 *git clone*

<img src="assets/img/F301_2_14.png" width="300"/>

*git clone* đã được giới thiệu ở phần 3.2.3. Chú ý rằng hiện tại ta làm việc với một thư mục không rỗng.

Ta sẽ *clone* *remote repository* vào 2 *local repository* mới `A` và `B`

Ta lấy địa chỉ clone từ

<img src="assets/img/F301_2_23.png" width="800"/>

``` sh
anybody@anywhere:~ $ cd workspace/

# Tạo thư mục cần thiết
anybody@anywhere:~/workspace $ mkdir A
anybody@anywhere:~/workspace $ mkdir B

# Đến thư mục A và clone
anybody@anywhere:~/workspace $ cd A

anybody@anywhere:~/workspace/A $ git clone https://github.com/inf301trainer/inf301.git
Cloning into 'inf301'...
remote: Enumerating objects: 5, done.
remote: Counting objects: 100% (5/5), done.
remote: Compressing objects: 100% (2/2), done.
remote: Total 5 (delta 0), reused 5 (delta 0), pack-reused 0
Unpacking objects: 100% (5/5), done.

# Đến thư mục B và clone
anybody@anywhere:~/workspace/A $ cd ..
anybody@anywhere:~/workspace $ cd B
anybody@anywhere:~/workspace/B $ git clone https://github.com/inf301trainer/inf301.git
Cloning into 'inf301'...
remote: Enumerating objects: 5, done.
remote: Counting objects: 100% (5/5), done.
remote: Compressing objects: 100% (2/2), done.
remote: Total 5 (delta 0), reused 5 (delta 0), pack-reused 0
Unpacking objects: 100% (5/5), done.
```

Khi clone, mọi branch của *remote repository* đều được update thành branch tương ứng của *local repository*.

#### 3.3.3 *git checkout*

**git checkout** là thao tác chuyển và tải code từ *branch* nào đó của *local git repository* vào working directory. Lúc này, working directory sẽ chứa code của *branch* tương ứng từ *local git repository*.

<img src="assets/img/F301_2_24.png" width="300"/>

Để kiểm tra branch của *local git repository* đang kết nối với local working directory, ta dùng **git branch**.

``` sh
# Đến thư mục git của nhóm A
anybody@anywhere:~/workspace/A 128 $ cd
anybody@anywhere:~ $ cd workspace/A/inf301/

# Ban đầu, chỉ có nhánh chính được liệt kê
anybody@anywhere:~/workspace/A/inf301 master ± git branch
* master

# Chuyển sang nhánh A
anybody@anywhere:~/workspace/A/inf301 master ± git checkout A
Branch 'A' set up to track remote branch 'A' from 'origin'.
Switched to a new branch 'A'

# Bây giờ có 2 nhánh, trong đó A (có dấu * ở đầu) là nhánh đang làm việc
anybody@anywhere:~/workspace/A/inf301 A ± git branch
* A
  master
```

#### 3.3.4 *git commit*, *git push*

Tương tự phần 3.2.4, 3.2.5, ta cập nhật một số thay đổi trong 2 nhánh:

Cho nhánh `A`:

``` sh
# Đây là trạng thái cũ (1 file)
anybody@anywhere:~ $ cd workspace/A/inf301/
anybody@anywhere:~/workspace/A/inf301 A ± ls
hello_world.py

# Thêm 1 file
anybody@anywhere:~/workspace/A/inf301 A ± echo "This is file A." > A.txt
anybody@anywhere:~/workspace/A/inf301 A* ± ls
A.txt  hello_world.py

# git add
anybody@anywhere:~/workspace/A/inf301 A* ± git add -A

# git config (cho lần đầu tiên)
anybody@anywhere:~/workspace/A/inf301 A(+0/-0) 128 ± git config --local user.name inf301trainer
anybody@anywhere:~/workspace/A/inf301 A(+0/-0) ± git config --local user.email a@gmail.com

# git commit
anybody@anywhere:~/workspace/A/inf301 A(+0/-0) ± git commit -m "Add file A"
[A 5352166] Add file A
 1 file changed, 1 insertion(+)
 create mode 100644 A.txt

# git push
anybody@anywhere:~/workspace/A/inf301 A(1) ± git push origin A
Username for 'https://github.com': inf301trainer
Password for 'https://inf301trainer@github.com':
Counting objects: 3, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (2/2), done.
Writing objects: 100% (3/3), 292 bytes | 292.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0)
To https://github.com/inf301trainer/inf301.git
   483530b..5352166  A -> A
```

Cho nhánh **B**

``` sh
anybody@anywhere:~ $ cd workspace/B/inf301/

# git checkout
anybody@anywhere:~/workspace/B/inf301 master* ± git checkout B
Branch 'B' set up to track remote branch 'B' from 'origin'.
Switched to a new branch 'B'

# Update files in working directory
anybody@anywhere:~/workspace/B/inf301 B* ± echo "This is file B." > B.txt
anybody@anywhere:~/workspace/B/inf301 B* ± ls
B.txt  hello_world.py

# git config
anybody@anywhere:~/workspace/B/inf301 B* ± git config --local user.name inf301trainer
anybody@anywhere:~/workspace/B/inf301 B* ± git config --local user.email b@gmail.com

# git add
anybody@anywhere:~/workspace/B/inf301 B* 1 ± git add -A
anybody@anywhere:~/workspace/B/inf301 B(+0/-0) ± git commit -m "Add file B"
[B 03f5eaa] Add file B
 1 file changed, 1 insertion(+)
 create mode 100644 B.txt

# git push
anybody@anywhere:~/workspace/B/inf301 B(1) ± git push origin B
Username for 'https://github.com': inf301trainer
Password for 'https://inf301trainer@github.com':
Counting objects: 3, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (2/2), done.
Writing objects: 100% (3/3), 286 bytes | 143.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0)
To https://github.com/inf301trainer/inf301.git
   483530b..03f5eaa  B -> B
```

Bây giờ, khi kiểm tra lại trên *remote repository*, ta thấy 2 nhánh `A`, `B` có các trạng thái khác nhau (một chứa `A.txt`, một chứa `B.txt`)

<img src="assets/img/F301_2_25.png" width="800"/>
<img src="assets/img/F301_2_26.png" width="800"/>

#### 3.3.5 Tạo *pull request*

<img src="assets/img/F301_2_30.png" width="300"/>

Giả sử nhóm *A* đã hoàn thành phần chỉnh sửa code và muốn đưa vào nhánh chính **master** của *main remote repository*. Thành viên sẽ tạo 1 **pull request** bằng cách click **New pull request**.

<img src="assets/img/F301_2_27.png" width="600"/>

Điền thông tin của **pull request**, nhấn **Create pull request**. Đảm bảo **base** là `master` và **compared** là `A` (trộn `A` vào `master`).

<img src="assets/img/F301_2_28.png" width="600"/>

Trong các dự án thực sự, ta cần sự kiểm tra của các thành viên khác.

<img src="assets/img/F301_2_29.png" width="600"/>

Muốn vậy ta chọn **Reviewers** ở cột bên phải và điền tên/username của thành viên kiểm tra.

Người kiểm tra sẽ **approve** (đồng ý), **decline** (từ chối) *pull request* hoặc bình luận. Nhóm `A` có thể tiếp tục sửa chữa code (`git add`, `git commit`, `git push`), sự thay đổi sẽ được cập nhật trên *pull request* cho đến khi được *approve*.

#### 3.3.6 Merge

<img src="assets/img/F301_2_30.png" width="300"/>

Một *pull request* được tán thành (*approve*) có thể được nhập/trộn (*merge*) vào nhánh chính `master`. Ví dụ, với nhánh `A`, click vào *pull request* và click vào **Merge pull request** -> **Confirm merge**.

Lúc này, kiểm tra lại nhánh `master`, ta thấy nó cũng bao gồm các *commit* trong `A`, cụ thể là cũng chứa file `A.txt`.

<img src="assets/img/F301_2_31.png" width="600"/>

#### 3.3.7 Cập nhật code sau các lần merge: *git fetch*, *git merge*, *git pull*

<img src="assets/img/F301_2_32.png" width="300"/>

Giả sử code trên *remote repository* đã được thay đổi sau khi merge branch `A` vào `master`. Bây giờ tại nhóm `B` có thể cập nhật phiên bản mới của code từ nhánh `A`, `master` cũng như `B`.

- **git fetch** là hành động tải code về *local repository*.

- **git merge** là hành động trộn code từ *local repository* (sau khi fetch) vào *working directory*.

``` sh
# Check trạng thái branch
anybody@anywhere:~/workspace/B/inf301 B 12s ± git branch
* B
  master

# git fetch
anybody@anywhere:~/workspace/B/inf301 B ± git fetch --all
Fetching origin
remote: Enumerating objects: 5, done.
remote: Counting objects: 100% (5/5), done.
remote: Compressing objects: 100% (3/3), done.
remote: Total 4 (delta 0), reused 3 (delta 0), pack-reused 0
Unpacking objects: 100% (4/4), done.
From https://github.com/inf301trainer/inf301
   483530b..5352166  A          -> origin/A
   483530b..4c3767c  master     -> origin/master

# git merge
anybody@anywhere:~/workspace/B/inf301 B ± git merge origin master
Merge made by the 'recursive' strategy.
 A.txt | 1 +
 1 file changed, 1 insertion(+)
 create mode 100644 A.txt

```

- **git pull** là tổng hợp hành động từ *git fetch* và *git merge*.

``` sh
# Dùng khi fetch và merge với phiên bản online của chính branch
git pull
# Dùng khi fetch và merge với phiên bản online của branch khác
git pull origin B
```

### 3.4 Minh họa nhánh phải của hình: tạo *forked remote repository* và làm việc với nó

#### 3.4.1 Fork

### 3.5 Một số lưu ý

#### 3.5.1 gitignore

#### 3.5.2 Khắc phục conflict

#### 3.5.3 Thay đổi phép truy cập

## 4 Bitbucket

Trong khóa học này, ta cần kết nối hệ quản lí phiên bản với nhiều công cụ khác như *Jenkins*, *Artifactory*. Vì vậy ta muốn hệ quản lí phiên bản của mình được cài đặt ở dạng *offline* (*cục bộ*). Ta có thể sử dụng phiên bản tương tự *Github* là *Gitlab*. Trong module này ta sẽ sử dụng một sản phẩm khác là ***Bitbucket***. (Tất nhiên, *Bitbucket* cũng là một sản phẩm online)

### 4.0 Tạo một tài khoản tại *Atlassian.com*

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

## Nguồn tư liệu

1. [confluence.atlassian.com/bitbucketserver](https://confluence.atlassian.com/bitbucketserver/bitbucket-server-installation-guide-867338382.html)
2. [hub.docker.com](https://hub.docker.com/r/atlassian/bitbucket-server/)
3. [subversion.apache.org](https://subversion.apache.org/)
4. [https://www.atlassian.com/git/tutorials/install-git](https://www.atlassian.com/git/tutorials/install-git)
