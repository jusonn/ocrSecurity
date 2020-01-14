## ocrSecurity's Github management rules
### Github에서 코딩작업을 할 때는 가급적 Issue, Branch, Wiki를 생성하여 작업 이력과 원인이 명확히 관리되도록 하며, 상세 절차는 아래를 따른다.

## Creating a branch from an issue
0. All issues are managed in our kanban board of our notion page
1. Create an issue in github. It will have "#+number" like "#1", "#2" automatically.
2. Remember this number and issue title you created.
3. Create an issue branch with the issue number and title in local pc. For example, "issue-1-feature/add-new-test-code", "issue-2-fix/test"
4. Checkout to new issue branch in local pc
5. Modify related codes
6. Create Wiki page to describe the revision and background knowledge.
8. Push the issue branch to github
9. Develop codes and add changes
10. Commit changes
11. Push the change to github
12. Create a pull request in github from issue branch to master branch
13. Review the PR and approve it
14. Merge PR
